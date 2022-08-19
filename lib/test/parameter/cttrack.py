from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
import importlib

def parameters(group_name:str, name: str, script_name: str):
	params = TrackerParams()
	prj_dir = env_settings().prj_dir
	save_dir = env_settings().save_dir
	# update default config from yaml file
	yaml_file = os.path.join(prj_dir, 'experiments', '%s' % name, '%s.yaml' % script_name)

	config_module = importlib.import_module('lib.config.{}.config'.format(name))
	config_module.update_config_from_file(yaml_file)
	cfg = config_module.cfg
	params.cfg = cfg
	print("test config: ", cfg)

	params.template_factor = cfg.DATA.TEMPLATE.FACTOR
	params.template_size = cfg.DATA.TEMPLATE.SIZE
	params.search_factor = cfg.DATA.SEARCH.FACTOR
	params.search_size = cfg.DATA.SEARCH.SIZE

	# Network checkpoint path
	# net_name = name.replace("_", "").upper()
	# epochid = 60
	# params.checkpoint = os.path.join(save_dir, "checkpoints", "train", name, script_name,
	#                                  '{}_ep{:04d}.pth.tar'.format(net_name, epochid))

	# params.checkpoint = os.path.join(save_dir, "checkpoints/train/maevit/vitpro_mix/MAECORNER_ep0080.pth.tar")
	params.checkpoint = '/home/uu201915762/workspace/CTTrack/VITPRO_online.pth.tar'

	# whether to save boxes from all queries
	params.save_all_boxes = False
	return params
