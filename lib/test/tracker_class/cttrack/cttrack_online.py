from lib.test.tracker_class.basetracker import BaseTracker
import torch
from lib.train.data.util.processing_utils import sample_target
# for debug
import cv2
import os
from lib.models.cttrack.cttrack import build_cttrack
from lib.test.utils.pre_processor import Preprocessor, Preprocessor_wo_mask
from lib.utils.box_ops import clip_box
from copy import deepcopy
from lib.utils.image import *

class CTTrackOnline(BaseTracker):
	def __init__(self, params, dataset_name):
		super(CTTrackOnline, self).__init__(params)
		network = build_cttrack(params.cfg)
		missing_keys, unexpected_keys = network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
		print("load checkpoint: ", self.params.checkpoint)
		print("missing keys:", missing_keys)
		print("unexpected keys:", unexpected_keys)
		self.cfg = params.cfg
		self.network = network.cuda()
		self.network.eval()
		self.preprocessor = Preprocessor_wo_mask()
		# self.preprocessor = Preprocessor()
		self.state = None
		# for debug
		self.debug = False
		self.frame_id = 0
		if self.debug:
			self.save_dir = "debug"
			if not os.path.exists(self.save_dir):
				os.makedirs(self.save_dir)
		# for save boxes from all queries
		self.save_all_boxes = params.save_all_boxes

		# Set the update interval
		DATASET_NAME = dataset_name.upper()
		if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
			self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
		else:
			self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
		print("Update interval is: ", self.update_intervals)

		if hasattr(self.cfg.TEST.ONLINE_SIZES, DATASET_NAME):
			self.online_size = self.cfg.TEST.ONLINE_SIZES[DATASET_NAME]
		else:
			self.online_size = 1
		print("online template size is: ", self.online_size)

	def initialize(self, image, info: dict):
		# forward the template once
		z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
													output_sz=self.params.template_size)
		template, template_amask = self.preprocessor.process(z_patch_arr, z_amask_arr)
		self.template = template
		self.template_amask = template_amask

		self.online_template = []
		self.online_template_amask = []
		self.online_template.append(deepcopy(template))
		self.online_template_amask.append(deepcopy(template_amask))
		self.max_pred_score = -1.0
		self.max_online_template = deepcopy(template)
		self.max_online_template_amask = deepcopy(template_amask)


		self.state = info['init_bbox']
		self.frame_id = 0
		if self.save_all_boxes:
			'''save all predicted boxes'''
			all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
			return {"all_boxes": all_boxes_save}

	def track(self, image, info: dict = None):
		H, W, _ = image.shape
		self.frame_id += 1
		x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
																output_sz=self.params.search_size)  # (x1, y1, w, h)
		search, search_amask = self.preprocessor.process(x_patch_arr, x_amask_arr)
		with torch.no_grad():
			# # run the transformer
			out_dict, _ = self.network.forward_test(template=self.template, online_template=self.online_template, search=search, run_score_head = True)

		pred_boxes = out_dict['pred_boxes'].view(-1, 4)
		pred_score = out_dict['pred_scores'].view(1).sigmoid().item()
		# Baseline: Take the mean of all pred boxes as the final result
		pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
		# get the final box result
		self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

		for idx, update_i in enumerate(self.update_intervals):
			if self.frame_id % update_i == 0 and pred_score > 0.9:
				online_patch_arr, _, online_amask_arr = sample_target(image, self.state, self.params.template_factor,
				                                                      output_sz=self.params.template_size)
				online_template, online_template_amask = self.preprocessor.process(online_patch_arr, online_amask_arr)
				self.online_template.append(online_template)
				self.online_template_amask.append(online_template_amask)
				if len(self.online_template) > self.online_size:
					self.online_template.pop(0)
					self.online_template_amask.pop(0)

			# for debug
		if self.debug:
			x1, y1, w, h = self.state
			image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
			save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
			cv2.imwrite(save_path, image_BGR)
		if self.save_all_boxes:
			'''save all predictions'''
			all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
			all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
			return {"target_bbox": self.state,
					"all_boxes": all_boxes_save}
		else:
			return {"target_bbox": self.state}

	def map_box_back(self, pred_box: list, resize_factor: float):
		cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
		cx, cy, w, h = pred_box
		half_side = 0.5 * self.params.search_size / resize_factor
		cx_real = cx + (cx_prev - half_side)
		cy_real = cy + (cy_prev - half_side)
		return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

	def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
		cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
		cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
		half_side = 0.5 * self.params.search_size / resize_factor
		cx_real = cx + (cx_prev - half_side)
		cy_real = cy + (cy_prev - half_side)
		return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

def get_tracker_class():
	return CTTrackOnline
