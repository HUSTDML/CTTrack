# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : vit_decoder.py
# Copyright (c) Skye-Song. All Rights Reserved

from xml.etree.ElementPath import xpath_tokenizer_re
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from lib.utils.image import *


def crop_search(image, target_bb, output_sz):
	""" Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area
	args:
		image: [B,C,H,W]
		target_bb - [B,4] = [x, y, w, h]
	"""

	B, C, H, W = image.shape
	output_images = []

	search_area_factor = 2.0
	for i in range(B):
		x, y, w, h = target_bb[i, :].detach().cpu().numpy()
		crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
		if crop_sz < 1:
			raise Exception('Too small bounding box in decoder.')
		x1 = round(x + 0.5 * w - crop_sz * 0.5)
		x2 = x1 + crop_sz

		y1 = round(y + 0.5 * h - crop_sz * 0.5)
		y2 = y1 + crop_sz

		x1_pad = max(0, -x1)
		x2_pad = max(x2 - W + 1, 0)

		y1_pad = max(0, -y1)
		y2_pad = max(y2 - H + 1, 0)

		output_images.append(
			F.interpolate(image[i, :, y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad].unsqueeze(0), (output_sz, output_sz), mode='bilinear'))

	return torch.cat(output_images, dim=0)

class SimMIMDecoder(nn.Module):
	def __init__(self, encoder_stride=16,mask_ratio=0.75,embed_dim=384,layer_nums=1):
		super().__init__()
		self.mask_ratio = mask_ratio
		# self.th=8
		# self.sh=20
		self.decoder_embed = nn.Conv2d(embed_dim, embed_dim, bias=True,kernel_size=1)
		self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.layers = nn.ModuleList([nn.Conv2d(embed_dim,embed_dim,kernel_size=1)\
            if i<layer_nums-1 else nn.Conv2d(embed_dim,encoder_stride**2*3,kernel_size=1) \
            for i in range(layer_nums)])
		# self.layers_s = nn.ModuleList([nn.Conv2d(embed_dim,embed_dim,kernel_size=1)\
        #     if i<layer_nums-1 else nn.Conv2d(embed_dim,embed_dim,kernel_size=1) \
        #     for i in range(layer_nums)])
		self.recover=nn.PixelShuffle(encoder_stride)

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
		x = rearrange(x, 'b c h w -> b (h w) c ').contiguous()
		x, mask = self.random_masking(x)

		H = int(math.sqrt(x.shape[1]))
		x = rearrange(x, 'b (h w) c -> b c h w', h=H).contiguous()
		# apply Transformer blocks
		for layer in self.layers:
			x = layer(x)
		
		x=self.recover(x)

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
		# input images = [b,3,h,w]
		pred, mask = self.forward_decoder(x)

		if gt_bboxes is not None:
			gt_bboxes = (gt_bboxes * pred.shape[-1]).to(torch.int64)
			pred = crop_search(image=pred, target_bb=gt_bboxes, output_sz=images.shape[-1])

		loss = self.forward_loss(imgs=images, pred=pred)
		return loss

def build_simmimdecoder(cfg):
	model = SimMIMDecoder(
		mask_ratio=0.75,
		embed_dim=384
	)
	return model
