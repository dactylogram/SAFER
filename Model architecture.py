'''
https://github.com/microsoft/Swin-Transformer
https://github.com/microsoft/SimMIM/blob/d3e29bcac950b83edc34ca33fe4404f38309052c/models/swin_transformer.py
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


### Hard coding for model parameters
patch_size = 4
embed_dim = 32
depths = [2, 2, 6, 2]
num_heads = [1, 2, 4, 8]
drop_path_rate = 0.15
window_size = 42
pool_scales = (12, 6, 4, 2) # 8, 12, 24, 48


### Conv helper ###
def conv_padding(input, kernel_size, stride, dilation=1):
    # convolution -> padding before convolution
    # l_out = (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    # => l_in / stride

    len_padding = dilation * (kernel_size-1) + 1 - stride

    pre_pad = len_padding // 2
    post_pad = len_padding - pre_pad
    input = torch.nn.functional.pad(input, (pre_pad, post_pad), 'constant', 0)
    return input

class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
    
    def forward(self, x):
        x = conv_padding(x, self.kernel_size, self.stride, self.dilation)
        x = self.conv1d(x)
        return x

class CBR(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, dilation=1, activation=True):
        super(CBR, self).__init__()
        self.C = Conv1D(n_in, n_out, kernel_size, stride, dilation)
        self.B = nn.BatchNorm1d(n_out)
        self.R = nn.GELU()
        self.activation = activation

    def forward(self, x):
        x=self.C(x)
        x=self.B(x)
        if self.activation == True:
            x=self.R(x)
        return x


### Swin transformer ###
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.view(-1, window_size, C)
    return windows

def window_reverse(windows, window_size, L):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Window size
        L (int): Length of signal

    Returns:
        x: (B, L, C)
    """
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    x = x.view(B, L, -1)
    return x

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The length of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * window_size-1, num_heads))  # 2*Wl-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_l = torch.arange(window_size)
        relative_position_index = coords_l[None,:] - coords_l[:,None] + window_size - 1
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # Wl,Wl,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wl, Wl
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        # x = x.view(B, L, C)

        # cyclic shift
        if self.shift_size > 0:
            img_mask = torch.zeros((1, L, 1))  # 1 L 1
            l_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for l in l_slices:
                img_mask[:, l, :] = cnt
                cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn_mask = attn_mask.to(x.device)
            shifted_x = torch.roll(x, shifts=(-self.shift_size), dims=(1))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size, C)  # nW*B, window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, L)  # B L C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size), dims=(1))
        else:
            x = shifted_x
        x = x.view(B, L, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, down_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim*down_ratio)
        self.down_ratio = down_ratio

        assert (down_ratio == 2) | (down_ratio == 4)

        if self.down_ratio > 2:
            self.reduction = nn.Linear(dim*down_ratio, dim*2, bias=False)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape

        # x = x.view(B, L, C)

        if self.down_ratio == 2:
            x0 = x[:, 0::2, :]  # B L/2 C
            x1 = x[:, 1::2, :]  # B L/2 C
            x = torch.cat([x0, x1], -1)  # B L/2 2*C
            x = x.view(B, -1, 2 * C)  # B L/2 2*C
            x = self.norm(x)
            # x = self.reduction(x)
        
        else:
            x0 = x[:, 0::4, :]  # B L/4 C
            x1 = x[:, 1::4, :]  # B L/4 C
            x2 = x[:, 2::4, :]  # B L/4 C
            x3 = x[:, 3::4, :]  # B L/4 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B L/4 4*C
            x = x.view(B, -1, 4 * C)  # B L/4 4*C
            x = self.norm(x)
            x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim//2, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=1, embed_dim=32, norm_layer=None):
        super().__init__()
        self.proj = Conv1D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, L = x.shape
        x = self.proj(x).transpose(1, 2)  # B Pl C
        if self.norm is not None:
            x = self.norm(x)
        return x

class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, patch_size=4, in_chans=1, num_classes=0,
                 embed_dim=32, depths=[2, 4, 6, 8], num_heads=[1, 2, 4, 8],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, init_patch_merge=True,
                 **kwargs):
        super().__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.init_patch_merge = init_patch_merge

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2**i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if i_layer != 0 else None)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, mask=None):
        if self.init_patch_merge:
            x = self.patch_embed(x)

        # Masking
        if mask is not None:
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)

            w = mask.unsqueeze(-1).type_as(mask_token)
            x = x * (1-w) + mask_token * w

        x = self.pos_drop(x)

        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x.transpose(1, 2))

        x = self.norm(x)  # B L C
        return x, features

    def forward(self, x, mask=None):
        x, features = self.forward_features(x, mask)
        return x, features

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}


### Upsampling ###
class PixelShuffle1D(nn.Module):
    # https://github.com/serkansulun/pytorch-pixelshuffle1d/blob/master/pixelshuffle1d.py
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x

class UpSamplePixelShuffle(nn.Module):
    def __init__(self, f_dim=256, out_channels=1, upscale=16):
        super().__init__()
        self.linear = Conv1D(in_channels=f_dim, out_channels=upscale*out_channels, kernel_size=1, stride=1)
        self.pixelshuffle = PixelShuffle1D(upscale_factor=upscale)
    
    def forward(self, x):
        B, L, C = x.shape
        x = torch.swapaxes(x, 1, 2)
        x = self.linear(x)
        x = self.pixelshuffle(x)
        return x


### Loss functions ###
class CELoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None):
        super().__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input, target):
        input = input.reshape(-1, input.shape[-1])
        target = target.flatten().long()

        ce_loss = self.CE(input, target)
        return ce_loss


### SigFormer pretrain ###
class SigFormer_pretrain(nn.Module):
    def __init__(self, patch_size=patch_size, in_chans=1, out_chans=1, num_classes=0,
                 embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=drop_path_rate,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 **kwargs):
        super().__init__()

        self.in_chans = in_chans
        self.patch_size = patch_size
        self.encoder = SwinTransformer(patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                                       embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size,
                                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                       attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                                       patch_norm=patch_norm)
        self.decoder = UpSamplePixelShuffle(f_dim=embed_dim*2**(len(depths)-1), out_channels=out_chans, upscale=patch_size*2**(len(depths)-1))
    
    def forward_features(self, x, mask=None):
        x_target = x.tanh()
        mid, features = self.encoder(x.tanh(), mask)
        x_recon = self.decoder(mid)
        # x_recon = x_recon.tanh()
        return x_target, x_recon

    def get_reconstruction(self, x):
        mid, features = self.encoder(x.tanh())
        x_recon = self.decoder(mid)
        return x_recon

    def calculate_loss(self, x_target, x_recon, mask=None):
        # loss_recon = F.l1_loss(x_target, x_recon, reduction='none')
        loss_recon = F.mse_loss(x_target, x_recon, reduction='none')

        if mask is not None:
            mask_ = mask.repeat_interleave(self.patch_size, 1).unsqueeze(1).contiguous()
            loss_mask = loss_recon[mask_.bool()].mean() / self.in_chans
            loss_clean = loss_recon[~mask_.bool()].mean() / self.in_chans
            # loss = loss_mask + loss_clean
            loss = loss_mask
        else:
            loss = loss_recon.mean() / self.in_chans

        self.loss_mask = loss_mask.item() if mask is not None else float('nan')
        self.loss_clean = loss_clean.item() if mask is not None else float('nan')

        return loss

    def forward(self, x, mask=None):
        x_target, x_recon = self.forward_features(x, mask)
        loss = self.calculate_loss(x_target, x_recon, mask)

        return loss


### SigFormer segmentation ###
class SigFormer_segmentation_rhythm(nn.Module):
    def __init__(self, patch_size=patch_size, in_chans=1, out_chans=2, num_classes=0,
                 embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=drop_path_rate,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 **kwargs):
        super().__init__()

        self.in_chans = in_chans
        self.patch_size = patch_size

        self.encoder = SwinTransformer(patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                                    embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size,
                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                    attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                                    patch_norm=patch_norm)
        self.decoder = UpSamplePixelShuffle(f_dim=embed_dim*2**(len(depths)-1), out_channels=out_chans, upscale=2**(len(depths)-1))
        # self.decoder = UpsamplePPM(in_channels=embed_dim*2**(len(depths)-1), out_channels=out_chans, upscale=2**(len(depths)-1))

    def forward_features(self, x, mask=None):
        out, features = self.encoder(x.tanh(), mask)
        seg = self.decoder(out)
        seg = seg.transpose(1, 2)

        return seg, out

    def calculate_loss(self, true, pred):
        criterion = CELoss()
        loss = criterion(pred.reshape(-1, pred.shape[-1]), true.flatten().long())
        return loss

    def forward(self, out, y):
        seg, _ = self.forward_features(out)
        loss = self.calculate_loss(y, seg)
        return loss


### Application ###
len_slice = 2688; mask_ratio = 1/3; batch_size = 128

# (1) Pseudocode for pretraining loss calculation
t_model_pretrain = SigFormer_pretrain()
x = torch.rand((batch_size, 1, len_slice)) # generated ECGs with fixed length

masks = [] # generate random masking
for i in range(batch_size):
    t_mask = np.zeros(len_slice // patch_size)
    len_mask_ecg = int(len_slice // patch_size * mask_ratio)
    t_mask[np.random.choice(range(t_mask.shape[0]), len_mask_ecg, replace=False)] = 1
    t_mask = torch.from_numpy(t_mask)
    masks.append(t_mask)
masks = torch.stack(masks, dim=0)

x_target, x_recon = t_model_pretrain.forward_features(x, mask=masks)
loss = t_model_pretrain.calculate_loss(x_target, x_recon, mask=masks)


# (2) Pseudocode for training loss calculation
t_model = SigFormer_segmentation_rhythm(out_chans=2)
x = torch.rand((batch_size, 1, len_slice)) # generated ECGs with fixed length
y = torch.randint(low=0, high=2, size=(batch_size, 1, len_slice // patch_size)) # corresponding labels

out, features = t_model.encoder(x.tanh())
seg, out = t_model.forward_features(x.float())
loss = t_model.calculate_loss(y, seg)
