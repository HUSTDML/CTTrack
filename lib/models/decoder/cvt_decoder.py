# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : vit_decoder.py
# Copyright (c) Skye-Song. All Rights Reserved

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D

from ..component.block import Block
from lib.utils.box_ops import box_xywh_to_cxywh, box_cxcywh_to_xyxy
from lib.utils.image import *

class deconv(nn.Module):
	def __init__(self, input_channel, output_channel, stride=2, kernel_size=3, padding=0):
		super().__init__()
		self.stride = stride
		self.conv = nn.Conv2d(input_channel, output_channel,
		                      kernel_size=kernel_size, stride=1, padding=padding)

	def forward(self, x):
		x = F.interpolate(x, scale_factor=self.stride, mode='bilinear',
		                  align_corners=True)
		return self.conv(x)


class DecoderLayer(nn.Module):
	def __init__(self, patch_size, patch_stride, patch_padding, decoder_embed_dim, conv_out_dim, decoder_num_heads,
	             decoder_depth, mlp_ratio, norm_layer=nn.LayerNorm):
		super().__init__()

		self.decoder_blocks = nn.ModuleList([
			Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
			for i in range(decoder_depth)])

		self.deconv = deconv(decoder_embed_dim, conv_out_dim, patch_stride, patch_size, patch_padding)

		self.norm = norm_layer(conv_out_dim) if norm_layer else None

	def forward(self, x):
		for blk in self.decoder_blocks:
			x = blk(x)

		H = int(math.sqrt(x.shape[1]))
		x = rearrange(x, 'b (h w) c -> b c h w', h=H).contiguous()
		x = self.deconv(x)

		if self.norm is not None:
			x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
			x = self.norm(x)

		return x


class CvtDecoder(nn.Module):
	def __init__(self, mask_ratio=0.75, embed_dim=384, patch_size=[3, 3, 7], patch_stride=[2, 2, 4],
	             patch_padding=[1, 1, 2], decoder_embed_dim=[384, 192, 64],
	             decoder_depth=[6, 1, 1], decoder_num_heads=[6, 3, 1],
	             mlp_ratio=[4., 4., 4.], pool_size=8, norm_layer=nn.LayerNorm, norm_pix_loss=False):
		super().__init__()
		self.mask_ratio = mask_ratio

		self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim[0], bias=True)
		self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim[0]))

		conv_out_dim = decoder_embed_dim + [3]
		conv_out_dim.pop(0)
		layer_nums = len(decoder_depth)
		self.layers = nn.ModuleList([
			DecoderLayer(patch_size=patch_size[i],
			             patch_stride=patch_stride[i],
			             patch_padding=patch_padding[i],
			             decoder_embed_dim=decoder_embed_dim[i],
			             conv_out_dim=conv_out_dim[i],
			             decoder_num_heads=decoder_num_heads[i],
			             decoder_depth=decoder_depth[i],
			             mlp_ratio=mlp_ratio[i],
			             norm_layer=norm_layer if i < layer_nums - 1 else None)
			for i in range(layer_nums)])

		self.norm_pix_loss = norm_pix_loss

		self.search_prroipool = PrRoIPool2D(pool_size, pool_size, spatial_scale=1.0)

		self.initialize_weights()

	def initialize_weights(self):
		# timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
		torch.nn.init.normal_(self.mask_token, std=.02)
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

	def random_masking(self, x):
		"""
		Perform per-sample random masking by per-sample shuffling.
		Per-sample shuffling is done by argsort random noise.
		x: [N, L, D], sequence
		"""
		N, L, D = x.shape  # batch, length, dim
		len_keep = int(L * (1 - self.mask_ratio))

		noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

		# sort noise for each sample
		ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
		ids_restore = torch.argsort(ids_shuffle, dim=1)

		# keep the first subset
		ids_keep = ids_shuffle[:, :len_keep]
		x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

		# generate the binary mask: 0 is keep, 1 is remove
		mask = torch.ones([N, L], device=x.device)
		mask[:, :len_keep] = 0
		# unshuffle to get the binary mask
		mask = torch.gather(mask, dim=1, index=ids_restore)

		# get the masked x
		mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x_keep.shape[1], 1)
		x_ = torch.cat([x_keep, mask_tokens], dim=1)  # no cls token
		x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

		return x_masked, mask

	def forward_decoder(self, x):

		# embed tokens
		x = self.decoder_embed(x)

		# append mask tokens to sequence
		x, mask = self.random_masking(x)

		# apply Transformer blocks
		for layer in self.layers:
			x = layer(x)

		return x, mask

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
		imgs, pred: [N, 3, H, W]
		"""
		if self.norm_pix_loss:
			mean = imgs.mean(dim=-1, keepdim=True)
			var = imgs.var(dim=-1, keepdim=True)
			imgs = (imgs - mean) / (var + 1.e-6) ** .5

		loss = (pred - imgs) ** 2
		loss = loss.mean()

		return loss

	def forward(self, x, images, gt_bboxes=None):
		# input x = [B,C,H,W]
		# input images = [b,3 h,w]

		if gt_bboxes is not None:
			x = self.crop_search_feat(x, gt_bboxes)

		x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
		pred, mask = self.forward_decoder(x)

		loss = self.forward_loss(imgs=images, pred=pred)
		return loss

	def crop_search_feat(self, search, gt_bboxes):
		# image: [B,C,H,W]
		# target_bb - [B,4] = [x, y, w, h]
		crop_bboxes = box_xywh_to_cxywh(gt_bboxes)
		crop_sz = torch.sqrt(gt_bboxes[:, 2] * gt_bboxes[:, 3]) * 2.0
		crop_sz = torch.clamp(crop_sz, min=0., max=1.)
		crop_bboxes[:, 2] = crop_bboxes[:, 3] = crop_sz

		crop_bboxes = crop_bboxes * search.shape[-1]
		crop_bboxes = box_cxcywh_to_xyxy(crop_bboxes.clone().view(-1, 4))
		batch_size = crop_bboxes.shape[0]
		batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(crop_bboxes.device)

		target_roi = torch.cat((batch_index, crop_bboxes), dim=1)
		search_box_feat = self.search_prroipool(search, target_roi)
		return search_box_feat


def build_cvtdecoder(cfg):
	pool_size = int(cfg.DATA.TEMPLATE.SIZE / 16)
	model = CvtDecoder(
		mask_ratio=0.75,
		embed_dim=384,
		patch_size=[3, 3, 7],
		patch_stride=[2, 2, 4],
		patch_padding=[1, 1, 3],
		decoder_embed_dim=[384, 192, 64],
		decoder_depth=[6, 1, 1],
		decoder_num_heads=[6, 3, 1],
		mlp_ratio=[4., 4., 4.],
		norm_layer=nn.LayerNorm,
		pool_size=pool_size,
		norm_pix_loss=False
		#
		# mask_ratio=cfg.MODEL.DECODER.MASK_RATIO,
		# patch_size=cfg.MODEL.BACKBONE.PATCHSIZE,
		# embed_dim=cfg.MODEL.BACKBONE.EMBEDDIM,
		# decoder_embed_dim=cfg.MODEL.DECODER.EMBEDDIM,
		# decoder_depth=cfg.MODEL.DECODER.DEPTH,
		# decoder_num_heads=cfg.MODEL.DECODER.NUMHEADS,
		# mlp_ratio=cfg.MODEL.DECODER.MLPRATIO,
		# norm_layer=nn.LayerNorm,
		# norm_pix_loss=False
	)
	return model
