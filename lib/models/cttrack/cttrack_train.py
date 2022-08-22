# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : vitcorner.py
# Copyright (c) Skye-Song. All Rights Reserved
import torch
from torch import nn
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_cxcywh_to_xywh
from einops import rearrange
from copy import deepcopy

from lib.models.backbone.vit import build_backbone as build_vit_backbone
from lib.models.boxhead.corner import build_box_head
from lib.models.decoder.mask_decoder import build_maskdecoder

class CTTrack_train(nn.Module):
	def __init__(self, backbone, box_head, decoder, cross_decoder = None):
		""" Initializes the model.
		"""
		super().__init__()
		self.backbone = backbone
		self.decoder = decoder
		self.box_head = box_head
		if cross_decoder is not None:
			self.cross_decoder = cross_decoder

	def forward(self, template, online_template, search, target_in_search, gt_bboxes,
	            t2t=False, t2s=False, s2t=False, s2s=False):
		# template, online_template: (b, c, h_t, w_t)
		# search: (b, c, h_s, w_s)
		# gt_box: (b,4)
		template_feat, online_template_feat, search_feat, cls_tokens = self.backbone(template, online_template,
		                                                                             search)
		out, outputs_coord_new = self.forward_box_head(search_feat, cls_tokens)

		loss = torch.tensor(0.0, dtype=torch.float32).to(gt_bboxes.device)
		if t2t:
			# template to template
			loss += self.decoder(template_feat, template)
			# online_template to online_template
			loss += self.decoder(online_template_feat, online_template)
		if t2s:
			# template to search
			loss += self.cross_decoder(template_feat, target_in_search)
			# online_template to search
			loss += self.cross_decoder(online_template_feat, target_in_search)
		if s2t:
			# search to template
			loss += self.cross_decoder(search_feat, template, gt_bboxes)
		if s2s:
			# search to search
			loss += self.decoder(search_feat, target_in_search, gt_bboxes)

		return loss, out, outputs_coord_new

	def powerful(self, search, cls_tokens):
		# search (B, C, H, W)
		# cls_tokens (B, 1, C)
		B, C, H, W = search.shape
		search = rearrange(search, 'b c h w-> b (h w) c').contiguous()  # (B, L, C)

		cls_tokens = cls_tokens.transpose(1, 2)  # (B, C, 1)
		power_map = torch.matmul(search, cls_tokens)  # (B, L, 1)

		opt = search.unsqueeze(-1) * power_map.unsqueeze(-2)  # (B, L, C, 1)*(B, L, 1, 1) = (B, L, C, 1)
		opt_feat = rearrange(opt.squeeze(-1), 'b (h w) c-> b c h w', h=H, w=W).contiguous()  # (B, C, H, W)
		return opt_feat

	def forward_box_head(self, search_feat, cls_tokens=None):
		"""
		:param search: (b, c, h, w)
		:return:
		"""
		if cls_tokens is not None:
			search_feat = self.powerful(search_feat, cls_tokens)

		b = search_feat.size(0)
		outputs_coord = box_xyxy_to_cxcywh(self.box_head(search_feat))
		outputs_coord_new = outputs_coord.view(b, 1, 4)
		out = {'pred_boxes': outputs_coord_new}
		return out, outputs_coord_new


def build_cttract_train(cfg):
	backbone = build_vit_backbone(cfg)

	box_head = build_box_head(cfg)  # a simple corner head

	decoder = build_maskdecoder(cfg)

	if getattr(cfg.TRAIN.RENEW, "TEMPLATE_TO_SEARCH", False) or getattr(cfg.TRAIN.RENEW, "SEARCH_TO_TEMPLATE", False):
		cross_decoder = deepcopy(decoder)
	else:
		cross_decoder = None

	model = CTTrack_train(
		backbone,
		box_head,
		decoder,
		cross_decoder
	)
	return model
