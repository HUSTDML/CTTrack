# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : gen_mask.py
# Copyright (c) Skye-Song. All Rights Reserved


import torch


def gen_attn_mask(template_num, template_dim, seq_num, seq_dim, search_num, search_dim):
	dims = template_num * template_dim + seq_num * seq_dim + search_num * search_dim
	mask = torch.zeros([dims, dims], dtype=torch.bool)
	mask[template_num * template_dim:template_num * template_dim + seq_num * seq_dim,
	0:template_num * template_dim] = True

	mask[0:template_num * template_dim,
	template_num * template_dim:template_num * template_dim + seq_num * seq_dim] = True

	return mask
