# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : params.py
# Copyright (c) Skye-Song. All Rights Reserved

def update_settings(settings, cfg):
	settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
	settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
	                               'search': cfg.DATA.SEARCH.FACTOR,
	                               'sequence': cfg.DATA.SEQUENCE.FACTOR
	                               }
	settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
	                      'search': cfg.DATA.SEARCH.SIZE,
	                      'sequence': cfg.DATA.SEQUENCE.SIZE}
	settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
	                                 'search': cfg.DATA.SEARCH.CENTER_JITTER,
	                                 'sequence': cfg.DATA.SEQUENCE.CENTER_JITTER}
	settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
	                                'search': cfg.DATA.SEARCH.SCALE_JITTER,
	                                'sequence': cfg.DATA.SEQUENCE.SCALE_JITTER}
	settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
	settings.print_stats = None
	settings.batchsize = cfg.TRAIN.BATCH_SIZE
	settings.patch_size = getattr(cfg.MODEL.BACKBONE, "PATCHSIZE", 0)
	settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
	settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
	settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
	settings.num_sequence = getattr(cfg.DATA.SEQUENCE, "NUMBER", 1)
	settings.target_in_search = getattr(cfg.DATA.TRAIN, "TARGET_IN_SEARCH", False)
	settings.need_mask_box = getattr(cfg.DATA.TRAIN, "NEED_MASK_BOX", False)

	renew = getattr(cfg.TRAIN, "RENEW", None)
	if renew is not None and not cfg.TRAIN.TRAIN_CLS:
		settings.loss_t2t = getattr(cfg.TRAIN.RENEW, "TEMPLATE_TO_TEMPLATE", False)
		settings.loss_t2s = getattr(cfg.TRAIN.RENEW, "TEMPLATE_TO_SEARCH", False)
		settings.loss_s2t = getattr(cfg.TRAIN.RENEW, "SEARCH_TO_TEMPLATE", False)
		settings.loss_s2s = getattr(cfg.TRAIN.RENEW, "SEARCH_TO_SEARCH", False)