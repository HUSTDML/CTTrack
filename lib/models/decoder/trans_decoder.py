# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : vit_decoder.py
# Copyright (c) Skye-Song. All Rights Reserved

import torch
import torch.nn as nn

from ..component.block import Block
from ..component.pos_embed import get_2d_sincos_pos_embed


class TransDecoder(nn.Module):
	def __init__(self, patch_size=16, num_patches=8 ** 2, embed_dim=1024, decoder_embed_dim=512,
	             decoder_depth=8, decoder_num_heads=16,
	             mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
		super().__init__()

		self.num_patches = num_patches
		self.patch_size = patch_size

		self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

		self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
		                                      requires_grad=False)  # fixed sin-cos embedding

		self.decoder_blocks = nn.ModuleList([
			Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
			for i in range(decoder_depth)])

		self.decoder_norm = norm_layer(decoder_embed_dim)
		self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3, bias=True)  # decoder to patch

		self.norm_pix_loss = norm_pix_loss

		self.initialize_weights()

	def initialize_weights(self):
		# initialize (and freeze) pos_embed by sin-cos embedding
		decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
		                                            int(self.num_patches ** .5), cls_token=False)
		self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

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

	def forward_decoder(self, x):
		# embed tokens
		x = self.decoder_embed(x)

		# add pos embed
		x = x + self.decoder_pos_embed

		# apply Transformer blocks
		for blk in self.decoder_blocks:
			x = blk(x)
		x = self.decoder_norm(x)

		# predictor projection
		x = self.decoder_pred(x)

		return x, None

	def patchify(self, imgs):
		"""
		imgs: (N, 3, H, W)
		x: (N, L, patch_size**2 *3)
		"""
		p = self.patch_size
		assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

		h = w = imgs.shape[2] // p
		x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
		x = torch.einsum('nchpwq->nhwpqc', x)
		x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
		return x

	def forward_loss(self, imgs, pred):
		"""
		imgs: [N, 3, H, W]
		pred: [N, L, p*p*3]
		"""
		target = self.patchify(imgs)
		if self.norm_pix_loss:
			mean = target.mean(dim=-1, keepdim=True)
			var = target.var(dim=-1, keepdim=True)
			target = (target - mean) / (var + 1.e-6) ** .5

		loss = (pred - target) ** 2
		loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

		return loss

	def forward(self, x, images):
		pred, _ = self.forward_decoder(x)  # [N, L, p*p*3]
		loss = self.forward_loss(imgs=images, pred=pred)
		return loss, pred

def trans_decoder():
	model = TransDecoder(
		patch_size=16, num_patches=8 ** 2, embed_dim=1024, decoder_embed_dim=512, decoder_depth=8,
		decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)
	return model


def build_transdecoder(cfg):
	num_patches = (cfg.DATA.TEMPLATE.SIZE // cfg.MODEL.BACKBONE.PATCHSIZE) ** 2
	model = TransDecoder(
		patch_size=cfg.MODEL.BACKBONE.PATCHSIZE,
		num_patches=num_patches,
		embed_dim=cfg.MODEL.BACKBONE.EMBEDDIM,
		decoder_embed_dim=cfg.MODEL.DECODER.EMBEDDIM,
		decoder_depth=cfg.MODEL.DECODER.DEPTH,
		decoder_num_heads=cfg.MODEL.DECODER.NUMHEADS,
		mlp_ratio=cfg.MODEL.DECODER.MLPRATIO,
		norm_layer=nn.LayerNorm,
		norm_pix_loss=False)
	return model
