# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : vit.py
# Copyright (c) Skye-Song. All Rights Reserved
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from ..component.patch_embed import PatchEmbed
from ..component.block import Block


class VisionTransformer(nn.Module):
	def __init__(self, patch_size=16, in_chans=3,
	             embed_dim=1024, depth=24, num_heads=16,
	             mlp_ratio=4., attention="Attention", drop_rate=0.,
	             attn_drop_rate=0., drop_path_rate=0.,
	             norm_layer=nn.LayerNorm, use_padding_mask=True, use_cls_token=False):
		super().__init__()

		self.patch_size = patch_size
		self.use_cls_token = use_cls_token
		if self.use_cls_token:
			self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
		self.pos_drop = nn.Dropout(p=drop_rate)

		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
		self.blocks = nn.ModuleList([
			Block(embed_dim,
			      num_heads,
			      mlp_ratio,
			      qkv_bias=True,
			      attention=attention,
			      drop=drop_rate,
			      attn_drop=attn_drop_rate,
			      drop_path=dpr[i],
			      norm_layer=norm_layer)
			for i in range(depth)])
		self.norm = norm_layer(embed_dim)

		self.use_padding_mask = use_padding_mask

		self.initialize_weights()

	def initialize_weights(self):
		w = self.patch_embed.proj.weight.data
		torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
		if self.use_cls_token:
			torch.nn.init.normal_(self.cls_token, std=.02)

		# initialize nn.Linear and nn.LayerNorm
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			# we use xavier_uniform following official JAX ViT:
			torch.nn.init.xavier_uniform_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	def deal_padding_mask(self, template_mask, online_template_mask, search_mask):
		if template_mask is None or online_template_mask is None or search_mask is None:
			return None

		template_mask = \
			F.interpolate(template_mask[None].float(), size=template_mask.shape[-1] // self.patch_size).to(torch.bool)[
				0]
		online_template_mask = \
			F.interpolate(online_template_mask[None].float(),
			              size=online_template_mask.shape[-1] // self.patch_size).to(
				torch.bool)[0]
		search_mask = \
			F.interpolate(search_mask[None].float(), size=search_mask.shape[-1] // self.patch_size).to(torch.bool)[0]

		merge_mask = torch.cat(
			[template_mask.flatten(1, 2), online_template_mask.flatten(1, 2), search_mask.flatten(1, 2)],
			dim=1)

		# merge_mask = merge_mask.to(torch.float16).mean(-1) >= 1
		return merge_mask

	def forward(self, template, online_template, search, template_mask=None, online_template_mask=None,
	            search_mask=None):
		"""
		       :param template: (batch, c, 128, 128)
		       :param online_template: (batch, c, 128, 128)
		       :param search: (batch, c, 320, 320)
		       :return:
       """
		t_B, t_C, t_H, t_W = template.size()
		s_B, s_C, s_H, s_W = search.size()
		t_H //= self.patch_size
		t_W //= self.patch_size
		s_H //= self.patch_size
		s_W //= self.patch_size

		# embed patches
		template = self.patch_embed(template)  # B,C,H,W -> B,HW,C
		online_template = self.patch_embed(online_template)
		search = self.patch_embed(search)

		# add pos embed w/o cls token
		# x = x + self.pos_embed[:, 1:, :]

		# apply Transformer blocks
		x = torch.cat([template, online_template, search], dim=1)
		if self.use_cls_token:
			cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
			x = torch.cat((cls_tokens, x), dim=1)

		x = self.pos_drop(x)

		if self.use_padding_mask:
			mask = self.deal_padding_mask(template_mask, online_template_mask, search_mask)
		else:
			mask = None

		for blk in self.blocks:
			x = blk(x, t_h=t_H, t_w=t_W, s_h=s_H, s_w=s_W, padding_mask=mask)
		x = self.norm(x)

		if self.use_cls_token:
			cls_tokens, template, online_template, search = torch.split(x, [1, t_H * t_W, t_H * t_W, s_H * s_W], dim=1)
		else:
			template, online_template, search = torch.split(x, [t_H * t_W, t_H * t_W, s_H * s_W], dim=1)
		template = rearrange(template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
		online_template = rearrange(online_template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
		search = rearrange(search, 'b (h w) c -> b c h w', h=s_H, w=s_W).contiguous()

		if self.use_cls_token:
			return template, online_template, search, cls_tokens
		return template, online_template, search, None

	def forward_test(self, template, online_template, search):
		"""
			   :param template: (batch, c, 128, 128)
			   :param online_template: [n * (batch, c, 128, 128)]
			   :param search: (batch, c, 320, 320)
			   :return:
	   """
		online_size = len(online_template)
		t_B, t_C, t_H, t_W = template.size()
		s_B, s_C, s_H, s_W = search.size()
		t_H //= self.patch_size
		t_W //= self.patch_size
		s_H //= self.patch_size
		s_W //= self.patch_size

		# embed patches
		template = self.patch_embed(template)  # B,C,H,W -> B,HW,C
		online_template = torch.cat([self.patch_embed(i) for i in online_template], dim=1)  # n*(B,C,H,W) -> B,n*HW,C
		search = self.patch_embed(search)

		# add pos embed w/o cls token
		# x = x + self.pos_embed[:, 1:, :]

		# apply Transformer blocks
		x = torch.cat([template, online_template, search], dim=1)
		if self.use_cls_token:
			cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
			x = torch.cat((cls_tokens, x), dim=1)

		x = self.pos_drop(x)

		for blk in self.blocks:
			x = blk(x, t_h=t_H, t_w=t_W, s_h=s_H, s_w=s_W, online_size = online_size)
		x = self.norm(x)

		if self.use_cls_token:
			cls_tokens, template, online_template, search = torch.split(x, [1, t_H * t_W, online_size * t_H * t_W, s_H * s_W],
			                                                            dim=1)
		else:
			template, online_template, search = torch.split(x, [t_H * t_W, online_size * t_H * t_W, s_H * s_W], dim=1)
		template = rearrange(template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
		online_template = rearrange(online_template, 'b (n h w) c -> b n c h w', n = online_size, h=t_H, w=t_W).contiguous()
		search = rearrange(search, 'b (h w) c -> b c h w', h=s_H, w=s_W).contiguous()

		if self.use_cls_token:
			return template, online_template, search, cls_tokens
		return template, online_template, search, None


def vit_base_patch16(**kwargs):
	model = VisionTransformer(
		patch_size=16, embed_dim=768, depth=12, num_heads=12,
		mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model


def vit_large_patch16(**kwargs):
	model = VisionTransformer(
		patch_size=16, embed_dim=1024, depth=24, num_heads=16,
		mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model


def vit_huge_patch14(**kwargs):
	model = VisionTransformer(
		patch_size=14, embed_dim=1280, depth=32, num_heads=16,
		mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model


def build_backbone(cfg):
	model = VisionTransformer(
		patch_size=cfg.MODEL.BACKBONE.PATCHSIZE,
		embed_dim=cfg.MODEL.BACKBONE.EMBEDDIM,
		depth=cfg.MODEL.BACKBONE.DEPTH,
		num_heads=cfg.MODEL.BACKBONE.NUMHEADS,
		mlp_ratio=cfg.MODEL.BACKBONE.MLPRATIO,
		attention=cfg.MODEL.BACKBONE.ATTENTION,
		drop_rate=cfg.MODEL.BACKBONE.DROP_RATE,
		attn_drop_rate=cfg.MODEL.BACKBONE.ATTN_DROP_RATE,
		drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE,
		use_padding_mask=cfg.MODEL.BACKBONE.USE_PADDING_MASK,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
		use_cls_token=cfg.MODEL.BACKBONE.USE_CLS_TOKEN
	)
	return model
