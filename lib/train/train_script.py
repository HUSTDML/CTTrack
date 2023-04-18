import os
import torch
import importlib

from torch.nn.functional import l1_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import BCEWithLogitsLoss

from lib.train.trainers import LTRTrainer
from lib.train.data.dataset_loader import build_seq_dataloaders
from lib.train.utils.optim_factory import get_optimizer_tt
from lib.train.utils.schedule_factory import get_schedule
from lib.train.utils.set_params import update_settings
from lib.train.actors import *

from lib.models.cttrack import build_cttrack,build_cttract_train

from lib.utils.box_ops import giou_loss

def run(settings):

	settings.description = 'Training script for transformer tracker'

	# update the default configs with config file
	if not os.path.exists(settings.cfg_file):
		raise ValueError("%s doesn't exist." % settings.cfg_file)
	config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
	cfg = config_module.cfg
	config_module.update_config_from_file(settings.cfg_file)
	if settings.local_rank in [-1, 0]:
		print("New configuration is shown below.")
		for key in cfg.keys():
			print("%s configuration:" % key, cfg[key])
			print('\n')

	# update settings based on cfg
	update_settings(settings, cfg)

	# Record the training log
	log_dir = os.path.join(settings.save_dir, 'logs')
	if settings.local_rank in [-1, 0]:
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
	settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

	# Build dataloaders
	loader_train, loader_val = build_seq_dataloaders(cfg, settings)
	# Create network

	if settings.script_name == "cttrack":
		net = build_cttract_train(cfg)
	elif settings.script_name == "cttrack_online":
		net = build_cttrack(cfg)
	else:
		raise ValueError("illegal script name")

	net.cuda()

	# wrap networks to distributed one
	if settings.local_rank != -1:
		net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
		settings.device = torch.device("cuda:%d" % settings.local_rank)
	else:
		settings.device = torch.device("cuda:0")

	# Loss functions and Actors
	if settings.script_name == "cttrack":
		objective = {'giou': giou_loss, 'l1': l1_loss}
		loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'renew': cfg.TRAIN.RENEW_WEIGHT}
		actor = CTTrackTrainActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
	elif  settings.script_name == "cttrack_online":
		objective = {'score': BCEWithLogitsLoss()}
		loss_weight = {'score': cfg.TRAIN.SCORE_WEIGHT}
		actor = CTTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings,
		                       run_score_head=True)
	else:
		raise ValueError("illegal script name")

	if cfg.TRAIN.DEEP_SUPERVISION:
		raise ValueError("Deep supervision is not supported now.")

	# Optimizer is for (1) choosing the training params and (2) training method (including setting the LR and momentum)
	optimizer = get_optimizer_tt(net, cfg)
	lr_scheduler = get_schedule(cfg, optimizer)

	use_amp = getattr(cfg.TRAIN, "AMP", True)
	trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

	# train process
	trainer.train(cfg.TRAIN.EPOCH, load_latest=False, fail_safe=True)