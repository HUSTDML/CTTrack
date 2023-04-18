import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import trunc_normal_

from ..component.mlp import MultiLayerMlp
from external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D


class ScoreDecoder(nn.Module):
	""" This is the base class for Transformer Tracking """

	def __init__(self, num_heads=6, pool_size=4, hidden_dim=384, mlp_layers=3):
		""" Initializes the model.
		"""
		super().__init__()
		self.num_heads = num_heads
		self.pool_size = pool_size
		self.score_head = MultiLayerMlp(hidden_dim, hidden_dim, 1, mlp_layers)
		self.scale = hidden_dim ** -0.5
		self.search_prroipool = PrRoIPool2D(pool_size, pool_size, spatial_scale=1.0)
		self.proj_q = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))
		self.proj_k = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))
		self.proj_v = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))

		self.proj = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))

		self.norm1 = nn.LayerNorm(hidden_dim)
		self.norm2 = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(2))

		self.score_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
		trunc_normal_(self.score_token, std=.02)

	def forward(self, search_feat, template_feat, search_box):
		"""
		:param search_box: with normalized coords. (x0, y0, x1, y1)
		:return:
		"""
		b, c, h, w = search_feat.shape
		search_box = search_box.clone() * w
		# bb_pool = box_cxcywh_to_xyxy(search_box.view(-1, 4))
		bb_pool = search_box.view(-1, 4)

		# Add batch_index to rois
		batch_size = bb_pool.shape[0]
		batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb_pool.device)
		target_roi = torch.cat((batch_index, bb_pool), dim=1)

		# decoder1: query for search_box feat
		# decoder2: query for template feat
		x = self.score_token.expand(b, -1, -1)
		x = self.norm1(x)
		search_box_feat = rearrange(self.search_prroipool(search_feat, target_roi), 'b c h w -> b (h w) c')
		template_feat = rearrange(template_feat, 'b c h w -> b (h w) c')
		kv_memory = [search_box_feat, template_feat]
		for i in range(2):
			q = rearrange(self.proj_q[i](x), 'b t (n d) -> b n t d', n=self.num_heads)
			k = rearrange(self.proj_k[i](kv_memory[i]), 'b t (n d) -> b n t d', n=self.num_heads)
			v = rearrange(self.proj_v[i](kv_memory[i]), 'b t (n d) -> b n t d', n=self.num_heads)

			attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
			attn = F.softmax(attn_score, dim=-1)
			x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
			x = rearrange(x, 'b h t d -> b t (h d)')  # (b, 1, c)
			x = self.proj[i](x)
			x = self.norm2[i](x)
		out_scores = self.score_head(x)  # (b, 1, 1)

		return out_scores

def build_score_head(cfg):
	pool_size = int(cfg.DATA.TEMPLATE.SIZE / cfg.MODEL.BACKBONE.PATCHSIZE)
	return ScoreDecoder(
		num_heads=cfg.MODEL.SCOREHEAD.NUMHEADS,
		pool_size=pool_size,
		hidden_dim=cfg.MODEL.BACKBONE.EMBEDDIM,
		mlp_layers=cfg.MODEL.SCOREHEAD.MLP_LAYERS
	)
