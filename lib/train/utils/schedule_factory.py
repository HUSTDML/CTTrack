# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : schedule_factory.py
# Copyright (c) Skye-Song. All Rights Reserved
import torch
import numpy as np
import math
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosLR(_LRScheduler):
	def __init__(self, optimizer, warmup_iters, epoch_max, warmup_factor=0.2, final_value_factor=0.1, last_epoch=-1):
		self.warmup_iters = warmup_iters
		self.T_max = epoch_max
		self.warmup_factor = warmup_factor
		self.final_value_factor = final_value_factor  # 最终的学习率(final_lr)是(base_lr)的倍数， final_lr = base_lr * final_value_factor
		super(WarmupCosLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		if self.last_epoch < self.warmup_iters:
			alpha = self.last_epoch / self.warmup_iters
			warmup_factor = self.warmup_factor * (1 - alpha) + alpha
			return [base_lr * warmup_factor
			        for base_lr in self.base_lrs]
		return [base_lr * self.final_value_factor + 0.5 * base_lr * (1 - self.final_value_factor) * (
				1 + math.cos(math.pi * (self.last_epoch - self.warmup_iters) / (self.T_max - self.warmup_iters)))
		        for base_lr in self.base_lrs]

# def _get_closed_form_lr(self):
# 	if self.last_epoch < self.warmup_iters:
# 		alpha = self.last_epoch / self.warmup_iters
# 		warmup_factor = self.warmup_factor * (1 - alpha) + alpha
# 		return [base_lr * warmup_factor
# 		        for base_lr in self.base_lrs]
# 	return [base_lr * self.final_factor + 0.5 * base_lr * (1 - self.final_value_factor) * (
# 			1 + math.cos(math.pi * self.last_epoch / self.T_max))
# 	        for base_lr in self.base_lrs]

class WarmUpStepLR(_LRScheduler):
	def __init__(self, optimizer, step_size, warmup_iters, warmup_factor=0.2, gamma=0.1, last_epoch=-1):
		self.step_size = step_size
		self.gamma = gamma
		self.warmup_iters = warmup_iters
		self.warmup_factor = warmup_factor
		super(WarmUpStepLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		if self.last_epoch < self.warmup_iters:
			alpha = self.last_epoch / self.warmup_iters
			warmup_factor = self.warmup_factor * (1 - alpha) + alpha
			return [base_lr * warmup_factor
			        for base_lr in self.base_lrs]
		elif (self.last_epoch - self.warmup_iters == 0) or (
				(self.last_epoch - self.warmup_iters) % self.step_size != 0):
			return [group['lr'] for group in self.optimizer.param_groups]
		else:
			return [group['lr'] * self.gamma
			        for group in self.optimizer.param_groups]

	def _get_closed_form_lr(self):
		if self.last_epoch < self.warmup_iters:
			alpha = self.last_epoch / self.warmup_iters
			warmup_factor = self.warmup_factor * (1 - alpha) + alpha
			return [base_lr * warmup_factor
			        for base_lr in self.base_lrs]
		return [base_lr * self.gamma ** ((self.last_epoch - self.warmup_iters) // self.step_size)
		        for base_lr in self.base_lrs]

# 采用pytorch官方的提供的调整学习率工具
def get_schedule(cfg, optimizer):
	if cfg.TRAIN.SCHEDULER.TYPE == 'step':
		lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
		                                               step_size=cfg.TRAIN.SCHEDULER.LR_STEP_SIZE,
		                                               gamma=cfg.TRAIN.SCHEDULER.LR_DROP_GAMMA)
	elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
		lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
		                                                    milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
		                                                    gamma=cfg.TRAIN.SCHEDULER.LR_DROP_GAMMA)
	elif cfg.TRAIN.SCHEDULER.TYPE == "warmup_step":
		lr_scheduler = WarmUpStepLR(optimizer,
		                            step_size=cfg.TRAIN.SCHEDULER.LR_STEP_SIZE,
		                            warmup_iters=cfg.TRAIN.SCHEDULER.WARMUP_EPOCH,
		                            warmup_factor=cfg.TRAIN.SCHEDULER.WARMUP_FACTOR,
		                            gamma=cfg.TRAIN.SCHEDULER.LR_DROP_GAMMA
		                            )

	elif cfg.TRAIN.SCHEDULER.TYPE == "warmup_cos":
		lr_scheduler = WarmupCosLR(optimizer,
		                           warmup_iters=cfg.TRAIN.SCHEDULER.WARMUP_EPOCH,
		                           epoch_max=cfg.TRAIN.EPOCH,
		                           warmup_factor=cfg.TRAIN.SCHEDULER.WARMUP_FACTOR,
		                           final_value_factor=cfg.TRAIN.SCHEDULER.WARMUP_FIANL_VALUE_FACTOR
		                           )
	else:
		raise ValueError("Unsupported scheduler")

	return lr_scheduler
