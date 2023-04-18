# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : optim_factory.py
# Copyright (c) Skye-Song. All Rights Reserved
import torch

from lib.train.admin.multigpu import is_multi_gpu

def get_optimizer(net, cfg):
	train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
	train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)

	if train_cls or train_score:
		print("Only training classification head. Learnable parameters are shown below.")
		param_dicts = [
			{"params": [p for n, p in net.named_parameters() if "cls_head" in n and p.requires_grad]}
		]

		for n, p in net.named_parameters():
			if "cls_head" not in n:
				p.requires_grad = False
			else:
				print(n)
	else:
		param_dicts = [
			{"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
			{
				"params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
				"lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
			},
		]

	opt_args = dict(lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
	if cfg.TRAIN.OPT_EPS is not None:
		opt_args['eps'] = cfg.TRAIN.OPT_EPS
	print("optimizer settings:", opt_args)

	opt_lower = cfg.TRAIN.OPTIMIZER.lower()
	if opt_lower == 'sgd' or opt_lower == 'nesterov':
		opt_args.pop('eps', None)
		optimizer = torch.optim.SGD(param_dicts, momentum=cfg.TRAIN.MOMENTUN, nesterov=True, **opt_args)
	elif opt_lower == 'momentum':
		opt_args.pop('eps', None)
		optimizer = torch.optim.SGD(param_dicts, momentum=cfg.TRAIN.MOMENTUN, nesterov=False, **opt_args)
	elif opt_lower == 'adam':
		optimizer = torch.optim.Adam(param_dicts, **opt_args)
	elif opt_lower == 'adamw':
		optimizer = torch.optim.AdamW(param_dicts, **opt_args)
	else:
		raise ValueError("Unsupported Optimizer")

	return optimizer


# --------------------------------------------------------------ViT Optimizer------------------------------------------------------------
def get_vit_parameter_groups(model, weight_decay=1e-5, layer_decay=0.75):
	parameter_group_names = {}
	parameter_groups = {}

	num_layers = len(model.blocks) + 1
	layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue  # frozen weights
		if param.ndim == 1 or name.endswith(".bias"):
			g_decay = "no_decay"
			this_weight_decay = 0.
		else:
			g_decay = "decay"
			this_weight_decay = weight_decay

		layer_id = get_layer_id_for_vit(name, num_layers)
		group_name = "layer_%d_%s" % (layer_id, g_decay)

		if group_name not in parameter_group_names:
			this_scale = layer_scales[layer_id]

			parameter_group_names[group_name] = {
				"weight_decay": this_weight_decay,
				"lr_scale": this_scale,
				"params": []
			}
			parameter_groups[group_name] = {
				"weight_decay": this_weight_decay,
				"lr_scale": this_scale,
				"params": []
			}
		parameter_group_names[group_name]["params"].append(name)
		parameter_groups[group_name]["params"].append(param)
	# print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
	return list(parameter_groups.values())


def get_layer_id_for_vit(name, num_layers):
	"""
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
	if name in ['cls_token', 'pos_embed']:
		return 0
	elif name.startswith('patch_embed'):
		return 0
	elif name.startswith('blocks'):
		return int(name.split('.')[1]) + 1
	else:
		return num_layers


def get_optimizer_tt(net, cfg):

	backbone_type = "vit"
	train_cls = cfg.TRAIN.TRAIN_CLS
	train_score = False
	train_decoder = False
	lr = cfg.TRAIN.LR
	weight_decay = cfg.TRAIN.WEIGHT_DECAY
	layer_decay = cfg.TRAIN.LAYER_DECAY

	if train_cls or train_score:
		print("Only training classification head. Learnable parameters are shown below.")
		param_dicts = [
			{"params": [p for n, p in net.named_parameters() if ("cls_head" in n or "score_head" in n) and p.requires_grad]}
		]

		for n, p in net.named_parameters():
			if ("cls_head" not in n) and ("score_head" not in n):
				p.requires_grad = False
			else:
				print(n)
	elif train_decoder:
		print("Only training decoder. Learnable parameters are shown below.")
		param_dicts = [
			{"params": [p for n, p in net.named_parameters() if "decoder" in n and p.requires_grad]}
		]

		for n, p in net.named_parameters():
			if "decoder" not in n:
				p.requires_grad = False
			else:
				print(n)
	elif backbone_type == "vit":
		net_backbone = net.module.backbone if is_multi_gpu(net) else net.backbone
		param_dicts = get_vit_parameter_groups(net_backbone, weight_decay=weight_decay,
		                                       layer_decay=layer_decay)
		param_dicts.append(
			{"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]})

		for pg in param_dicts:
			if "lr_scale" in pg:
				pg["lr"] = lr * pg["lr_scale"]
			else:
				pg["lr"] = lr

		print("vit optimizer checked!")
	elif backbone_type == "cvt":
		param_dicts = [
			{"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
			{
				"params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
				"lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
			},
		]
		print("cvt optimizer checked!")
	else:
		raise ValueError("wrong training type")

	opt_args = dict(lr=lr, weight_decay=weight_decay)
	if cfg.TRAIN.OPT_EPS is not None:
		opt_args['eps'] = cfg.TRAIN.OPT_EPS
	print("optimizer settings:", opt_args)

	opt_lower = cfg.TRAIN.OPTIMIZER.lower()
	if opt_lower == 'sgd' or opt_lower == 'nesterov':
		opt_args.pop('eps', None)
		optimizer = torch.optim.SGD(param_dicts, momentum=cfg.TRAIN.MOMENTUN, nesterov=True, **opt_args)
	elif opt_lower == 'momentum':
		opt_args.pop('eps', None)
		optimizer = torch.optim.SGD(param_dicts, momentum=cfg.TRAIN.MOMENTUN, nesterov=False, **opt_args)
	elif opt_lower == 'adam':
		optimizer = torch.optim.Adam(param_dicts, **opt_args)
	elif opt_lower == 'adamw':
		optimizer = torch.optim.AdamW(param_dicts, **opt_args)
	else:
		raise ValueError("Unsupported Optimizer")

	return optimizer
