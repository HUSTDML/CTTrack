# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : vitcorner.py
# Copyright (c) Skye-Song. All Rights Reserved
from torch import nn
from einops import rearrange
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.models.backbone.vit import build_backbone
from lib.models.boxhead.corner import build_box_head
from lib.models.component.mlp import MultiLayerMlp
from lib.utils.image import *


class CTTrack(nn.Module):
	def __init__(self, backbone, box_head, score_head=None):
		""" Initializes the model.
		"""
		super().__init__()
		self.backbone = backbone
		self.box_head = box_head
		self.score_head = score_head

	def forward(self, template, online_template, search, run_score_head=False):
		# input image: (b, c, h, w)
		template, online_template, search, cls_tokens = self.backbone(template, online_template, search)
		# Forward the corner head
		return self.forward_head(search, cls_tokens, run_score_head)

	def forward_test(self, template, online_template, search, run_score_head=False):
		# template and serach: (b, c, h, w)
		# online_template n*(b, c, h, w)
		template, online_template, search, cls_tokens = self.backbone.forward_test(template, online_template, search)
		# Forward the corner head
		return self.forward_head(search, cls_tokens, run_score_head)

	def powerful(self, search, cls_tokens):
		# search (B, C, H, W)
		# cls_tokens (B, 1, C)
		B,C,H,W = search.shape
		search = rearrange(search, 'b c h w-> b (h w) c').contiguous() #(B, L, C)

		cls_tokens = cls_tokens.transpose(1, 2)  # (B, C, 1)
		power_map = torch.matmul(search, cls_tokens)  # (B, L, 1)

		opt = search.unsqueeze(-1) * power_map.unsqueeze(-2) #(B, L, C, 1)*(B, L, 1, 1) = (B, L, C, 1)
		opt_feat = rearrange(opt.squeeze(-1), 'b (h w) c-> b c h w', h=H, w=W).contiguous() # (B, C, H, W)
		return opt_feat

	def forward_head(self, search, cls_tokens, run_score_head=True):
		"""
		:param search: (b, c, h, w), reg_mask: (b, h, w)
		:return:
		"""
		out_dict = {}
		opt_feat = self.powerful(search,cls_tokens)
		out_dict_box, outputs_coord = self.forward_box_head(opt_feat)
		out_dict.update(out_dict_box)
		if run_score_head and self.score_head is not None:
			out_dict.update({'pred_scores': self.score_head(cls_tokens)})
		return out_dict, outputs_coord

	def forward_box_head(self, search):
		"""
		:param search: (b, c, h, w)
		:return:
		"""
		b = search.size(0)
		outputs_coord = box_xyxy_to_cxcywh(self.box_head(search))
		outputs_coord_new = outputs_coord.view(b, 1, 4)
		out = {'pred_boxes': outputs_coord_new}
		return out, outputs_coord_new

def build_cttrack(cfg):
	backbone = build_backbone(cfg)  # backbone without positional encoding and attention mask
	box_head = build_box_head(cfg)  # a simple corner head
	score_head = None
	if cfg.MODEL.SCOREHEAD.EXIST:
		score_head = MultiLayerMlp(cfg.MODEL.BACKBONE.EMBEDDIM, cfg.MODEL.BACKBONE.EMBEDDIM, 1, 3)
	model = CTTrack(
		backbone,
		box_head,
		score_head
	)
	return model
