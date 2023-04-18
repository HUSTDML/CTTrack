from easydict import EasyDict as edict
import yaml

"""
TRAIN MAIN
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
# cfg.MODEL.NLAYER_HEAD = 3
# cfg.MODEL.HIDDEN_DIM = 256
# cfg.MODEL.NUM_OBJECT_QUERIES = 1

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.PATCHSIZE = 16
cfg.MODEL.BACKBONE.EMBEDDIM = 768
cfg.MODEL.BACKBONE.DEPTH = 12
cfg.MODEL.BACKBONE.NUMHEADS = 12
cfg.MODEL.BACKBONE.MLPRATIO = 4
cfg.MODEL.BACKBONE.ATTENTION = "Attention"
cfg.MODEL.BACKBONE.DROP_RATE = 0.0
cfg.MODEL.BACKBONE.ATTN_DROP_RATE = 0.0
cfg.MODEL.BACKBONE.DROP_PATH_RATE = 0.1
cfg.MODEL.BACKBONE.USE_PADDING_MASK = False
cfg.MODEL.BACKBONE.USE_CLS_TOKEN = True

# MODEL.BOXHEAD
cfg.MODEL.BOXHEAD = edict()
cfg.MODEL.BOXHEAD.IN_DIM = 768
cfg.MODEL.BOXHEAD.HEAD_DIM = 384
cfg.MODEL.BOXHEAD.FREEZE_BN = True

# MODEL.SCOREHEAD
cfg.MODEL.SCOREHEAD = edict()
cfg.MODEL.SCOREHEAD.EXIST = True
cfg.MODEL.SCOREHEAD.NUMHEADS = 6
cfg.MODEL.SCOREHEAD.MLP_LAYERS = 3

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.TRAIN_CLS = True
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.EPOCH = 80
cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.NUM_WORKER = 8

# TRAIN.OPTIMIZER
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.LAYER_DECAY = 0.75
cfg.TRAIN.OPT_EPS = 1e-10
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1

# no use
cfg.TRAIN.DEEP_SUPERVISION = False

# TRAIN.LOSS
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.SCORE_WEIGHT = 1.0

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.LR_DROP_EPOCH = 400  # no use for "warmup_cos"
cfg.TRAIN.SCHEDULER.TYPE = "warmup_cos"
cfg.TRAIN.SCHEDULER.WARMUP_EPOCH = 5
cfg.TRAIN.SCHEDULER.WARMUP_FACTOR = 0.2
cfg.TRAIN.SCHEDULER.WARMUP_FIANL_VALUE_FACTOR = 0.1
# cfg.TRAIN.SCHEDULER.TYPE = "step"
# cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
# cfg.TRAIN.SCHEDULER.LR_STEP_SIZE = 100
# cfg.TRAIN.SCHEDULER.LR_STEP_GAMMA = 0.75

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = [200]
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.NEED_MASK_BOX = False
cfg.DATA.TRAIN.TARGET_IN_SEARCH = False
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1  # number of search frames for multiple frames training
cfg.DATA.SEARCH.SIZE = 384
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 2
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0
# DATA.SEQUENCE
cfg.DATA.SEQUENCE = edict()
cfg.DATA.SEQUENCE.NUMBER = 2
cfg.DATA.SEQUENCE.SIZE = 256
cfg.DATA.SEQUENCE.FACTOR = 3.0
cfg.DATA.SEQUENCE.CENTER_JITTER = 0.5
cfg.DATA.SEQUENCE.SCALE_JITTER = 0.1

# TEST
cfg.TEST = edict()
cfg.TEST.UPDATE_INTERVALS = edict()
cfg.TEST.UPDATE_INTERVALS.LASOT = [200]
cfg.TEST.UPDATE_INTERVALS.GOT10K_TEST = [200]
cfg.TEST.UPDATE_INTERVALS.TRACKINGNET = [200]
cfg.TEST.UPDATE_INTERVALS.VOT20 = [25]
cfg.TEST.UPDATE_INTERVALS.VOT20LT = [200]
cfg.TEST.UPDATE_INTERVALS.OTB = [100]
cfg.TEST.UPDATE_INTERVALS.UAV = [100]

cfg.TEST.ONLINE_SIZES = edict()
cfg.TEST.ONLINE_SIZES.LASOT = 2
cfg.TEST.ONLINE_SIZES.UAV = 1
cfg.TEST.ONLINE_SIZES.OTB = 3
cfg.TEST.ONLINE_SIZES.GOT10K_TEST = 2
cfg.TEST.ONLINE_SIZES.TRACKINGNET = 1
cfg.TEST.ONLINE_SIZES.VOT20 = 5


def _edict2dict(dest_dict, src_edict):
	if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
		for k, v in src_edict.items():
			if not isinstance(v, edict):
				dest_dict[k] = v
			else:
				dest_dict[k] = {}
				_edict2dict(dest_dict[k], v)
	else:
		return


def gen_config(config_file):
	cfg_dict = {}
	_edict2dict(cfg_dict, cfg)
	with open(config_file, 'w') as f:
		yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
	if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
		for k, v in exp_cfg.items():
			if k in base_cfg:
				if not isinstance(v, dict):
					base_cfg[k] = v
				else:
					_update_config(base_cfg[k], v)
			else:
				raise ValueError("{} not exist in config.py".format(k))
	else:
		return


def update_config_from_file(filename):
	exp_config = None
	with open(filename) as f:
		exp_config = edict(yaml.safe_load(f))
		_update_config(cfg, exp_config)
