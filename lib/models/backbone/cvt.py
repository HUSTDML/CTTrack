# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : vit.py
# Copyright (c) Skye-Song. All Rights Reserved
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from torch._six import container_abcs

from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_

from functools import partial
from itertools import repeat

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
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


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias

class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 freeze_bn=False,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token
        if freeze_bn:
            conv_proj_post_norm = FrozenBatchNorm2d
        else:
            conv_proj_post_norm = nn.BatchNorm2d

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method, conv_proj_post_norm
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method, conv_proj_post_norm
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method, conv_proj_post_norm
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method,
                          norm):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', norm(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, t_h, t_w, s_h, s_w):
        template, online_template, search = torch.split(x, [t_h * t_w, t_h * t_w, s_h * s_w], dim=1)
        template = rearrange(template, 'b (h w) c -> b c h w', h=t_h, w=t_w).contiguous()
        online_template = rearrange(online_template, 'b (h w) c -> b c h w', h=t_h, w=t_w).contiguous()
        search = rearrange(search, 'b (h w) c -> b c h w', h=s_h, w=s_w).contiguous()

        if self.conv_proj_q is not None:
            t_q = self.conv_proj_q(template)
            ot_q = self.conv_proj_q(online_template)
            s_q = self.conv_proj_q(search)
            q = torch.cat([t_q, ot_q, s_q], dim=1)
        else:
            t_q = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            ot_q = rearrange(online_template, 'b c h w -> b (h w) c').contiguous()
            s_q = rearrange(search, 'b c h w -> b (h w) c').contiguous()
            q = torch.cat([t_q, ot_q, s_q], dim=1)

        if self.conv_proj_k is not None:
            t_k = self.conv_proj_k(template)
            ot_k = self.conv_proj_k(online_template)
            s_k = self.conv_proj_k(search)
            k = torch.cat([t_k, ot_k, s_k], dim=1)
        else:
            t_k = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            ot_k = rearrange(online_template, 'b c h w -> b (h w) c').contiguous()
            s_k = rearrange(search, 'b c h w -> b (h w) c').contiguous()
            k = torch.cat([t_k, ot_k, s_k], dim=1)

        if self.conv_proj_v is not None:
            t_v = self.conv_proj_v(template)
            ot_v = self.conv_proj_v(online_template)
            s_v = self.conv_proj_v(search)
            v = torch.cat([t_v, ot_v, s_v], dim=1)
        else:
            t_v = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            ot_v = rearrange(online_template, 'b c h w -> b (h w) c').contiguous()
            s_v = rearrange(search, 'b c h w -> b (h w) c').contiguous()
            v = torch.cat([t_v, ot_v, s_v], dim=1)

        return q, k, v

    def forward_conv_test(self, x, s_h, s_w):
        search = x
        search = rearrange(search, 'b (h w) c -> b c h w', h=s_h, w=s_w).contiguous()

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(search)
        else:
            q = rearrange(search, 'b c h w -> b (h w) c').contiguous()

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(search)
        else:
            k = rearrange(search, 'b c h w -> b (h w) c').contiguous()
        k = torch.cat([self.t_k, self.ot_k, k], dim=1)

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(search)
        else:
            v = rearrange(search, 'b c h w -> b (h w) c').contiguous()
        v = torch.cat([self.t_v, self.ot_v, v], dim=1)

        return q, k, v


    def forward(self, x, t_h, t_w, s_h, s_w):
        """
        Asymmetric mixed attention.
        """
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, t_h, t_w, s_h, s_w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads).contiguous()
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads).contiguous()
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads).contiguous()


        # attn_score = torch.einsum('bhld,bhtd->bhlt', [q, k]) * self.scale
        # attn = F.softmax(attn_score, dim=-1)
        # attn = self.attn_drop(attn)
        # x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        # x = rearrange(x, 'b h t d -> b t (h d)')

        # x = self.proj(x)
        # x = self.proj_drop(x)

        # return x

        # ### Attention!: k/v compression，1/4 of q_size（conv_stride=2）
        q_t,q_ot,q_s = torch.split(q, [t_h * t_w ,t_h * t_w , s_h * s_w], dim=2)
        # k_t, k_ot, k_s = torch.split(k, [t_h*t_w//4, t_h*t_w//4, s_h*s_w//4], dim=2)
        # v_t, v_ot, v_s = torch.split(v, [t_h * t_w // 4, t_h * t_w // 4, s_h * s_w // 4], dim=2)
        k_mt, k_s = torch.split(k, [((t_h + 1) // 2) ** 2 * 2, s_h * s_w // 4], dim=2)
        v_mt, v_s = torch.split(v, [((t_h + 1) // 2) ** 2 * 2, s_h * s_w // 4], dim=2)

        # template attention
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q_t, k_mt]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        x_t = torch.einsum('bhlt,bhtv->bhlv', [attn, v_mt])
        x_t = rearrange(x_t, 'b h t d -> b t (h d)')


        # online template attention
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q_ot, k_mt]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        x_ot = torch.einsum('bhlt,bhtv->bhlv', [attn, v_mt])
        x_ot = rearrange(x_ot, 'b h t d -> b t (h d)')

        # search region attention
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q_s, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        x_s = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x_s = rearrange(x_s, 'b h t d -> b t (h d)')

        x = torch.cat([x_t,x_ot, x_s], dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def forward_test(self, x, s_h, s_w):
        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q_s, k, v = self.forward_conv_test(x, s_h, s_w)

        q_s = rearrange(self.proj_q(q_s), 'b t (h d) -> b h t d', h=self.num_heads).contiguous()
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads).contiguous()
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads).contiguous()

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q_s, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        x_s = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x_s = rearrange(x_s, 'b h t d -> b t (h d)').contiguous()

        x = x_s

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


    def set_online(self, x, t_h, t_w):
        template = x[:, :t_h * t_w]  # 1, 1024, c
        online_template = x[:, t_h * t_w:]  # 1, b*1024, c
        template = rearrange(template, 'b (h w) c -> b c h w', h=t_h, w=t_w).contiguous()
        online_template = rearrange(online_template.squeeze(0), '(b h w) c -> b c h w', h=t_h, w=t_w).contiguous()  # b, c, 32, 32
        if self.conv_proj_q is not None:
            t_q = self.conv_proj_q(template)
            ot_q = self.conv_proj_q(online_template).flatten(end_dim=1).unsqueeze(0)
        else:
            t_q = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            ot_q = rearrange(online_template, 'b c h w -> (b h w) c').contiguous().unsqueeze(0)
        q = torch.cat([t_q, ot_q], dim=1)

        if self.conv_proj_k is not None:
            self.t_k = self.conv_proj_k(template)
            self.ot_k = self.conv_proj_k(online_template).flatten(end_dim=1).unsqueeze(0)
        else:
            self.t_k = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            self.ot_k = rearrange(online_template, 'b c h w -> (b h w) c').contiguous().unsqueeze(0)
        k = torch.cat([self.t_k, self.ot_k], dim=1)

        if self.conv_proj_v is not None:
            self.t_v = self.conv_proj_v(template)
            self.ot_v = self.conv_proj_v(online_template).flatten(end_dim=1).unsqueeze(0)
        else:
            self.t_v = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            self.ot_v = rearrange(online_template, 'b c h w -> (b h w) c').contiguous().unsqueeze(0)
        v = torch.cat([self.t_v, self.ot_v], dim=1)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads).contiguous()
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads).contiguous()
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads).contiguous()

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)').contiguous()

        # x = torch.cat([self.x_t, self.x_ot], dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 freeze_bn=False,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop, freeze_bn=freeze_bn,
            **kwargs
        )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, t_h, t_w, s_h, s_w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, t_h, t_w, s_h, s_w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def forward_test(self, x, s_h, s_w):
        res = x

        x = self.norm1(x)
        attn = self.attn.forward_test(x, s_h, s_w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def set_online(self, x, t_h, t_w):
        res = x
        x = self.norm1(x)
        attn = self.attn.set_online(x, t_h, t_w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 freeze_bn=False,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            )
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    freeze_bn=freeze_bn,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, template, online_template, search):
        """
        :param template: (batch, c, 128, 128)
        :param search: (batch, c, 320, 320)
        :return:
        """
        # x = self.patch_embed(x)
        # B, C, H, W = x.size()
        template = self.patch_embed(template)
        online_template = self.patch_embed(online_template)
        t_B, t_C, t_H, t_W = template.size()
        search = self.patch_embed(search)
        s_B, s_C, s_H, s_W = search.size()

        template = rearrange(template, 'b c h w -> b (h w) c').contiguous()
        online_template = rearrange(online_template, 'b c h w -> b (h w) c').contiguous()
        search = rearrange(search, 'b c h w -> b (h w) c').contiguous()
        x = torch.cat([template, online_template, search], dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, t_H, t_W, s_H, s_W)

        # if self.cls_token is not None:
        #     cls_tokens, x = torch.split(x, [1, H*W], 1)
        template, online_template, search = torch.split(x, [t_H*t_W, t_H*t_W, s_H*s_W], dim=1)
        template = rearrange(template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        online_template = rearrange(online_template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        search = rearrange(search, 'b (h w) c -> b c h w', h=s_H, w=s_W).contiguous()

        return template, online_template, search

    def forward_test(self, search):
        # x = self.patch_embed(x)
        # B, C, H, W = x.size()
        search = self.patch_embed(search)
        s_B, s_C, s_H, s_W = search.size()

        search = rearrange(search, 'b c h w -> b (h w) c').contiguous()
        x = search
        # x = torch.cat([template, search], dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk.forward_test(x, s_H, s_W)

        # if self.cls_token is not None:
        #     cls_tokens, x = torch.split(x, [1, H*W], 1)
        # template, search = torch.split(x, [t_H*t_W, s_H*s_W], dim=1)
        search = x
        search = rearrange(search, 'b (h w) c -> b c h w', h=s_H, w=s_W)

        return search

    def set_online(self, template, online_template):
        template = self.patch_embed(template)
        online_template = self.patch_embed(online_template)
        t_B, t_C, t_H, t_W = template.size()

        template = rearrange(template, 'b c h w -> b (h w) c').contiguous()
        online_template = rearrange(online_template, 'b c h w -> (b h w) c').unsqueeze(0).contiguous()
        # 1, 1024, c
        # 1, b*1024, c
        # print(template.shape, online_template.shape)
        x = torch.cat([template, online_template], dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk.set_online(x, t_H, t_W)

        # if self.cls_token is not None:
        #     cls_tokens, x = torch.split(x, [1, H*W], 1)
        template = x[:, :t_H*t_W]
        online_template = x[:, t_H*t_W:]
        template = rearrange(template, 'b (h w) c -> b c h w', h=t_H, w=t_W)
        online_template = rearrange(online_template.squeeze(0), '(b h w) c -> b c h w', h=t_H, w=t_W)

        return template, online_template


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 # num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        # self.num_classes = num_classes

        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
                'freeze_bn': spec['FREEZE_BN'],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        # dim_embed = spec['DIM_EMBED'][-1]
        # self.norm = norm_layer(dim_embed)
        # self.cls_token = spec['CLS_TOKEN'][-1]

        # # Classifier head
        # self.head = nn.Linear(dim_embed, 1000)
        # trunc_normal_(self.head.weight, std=0.02)

    def forward(self, template, online_template, search):
        """
        :param template: (b, 3, 128, 128)
        :param search: (b, 3, 320, 320)
        :return:
        """
        # template = template + self.template_emb
        # search = search + self.search_emb
        for i in range(self.num_stages):
            template, online_template, search = getattr(self, f'stage{i}')(template, online_template, search)

        return template, online_template, search

    def forward_test(self, search):
        for i in range(self.num_stages):
            search = getattr(self, f'stage{i}').forward_test(search)
        return search

    def set_online(self, template, online_template):
        for i in range(self.num_stages):
            template, online_template = getattr(self, f'stage{i}').set_online(template, online_template)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def build_backbone(config, **kwargs):
    msvit_spec = config.MODEL.BACKBONE
    msvit = ConvolutionalVisionTransformer(
        in_chans=3,
        act_layer=QuickGELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
        init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
        spec=msvit_spec
    )

    return msvit
