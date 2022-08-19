from lib.test.tracker_class.basetracker import BaseTracker
import torch
from lib.train.data.util.processing_utils import sample_target
# for debug
import cv2
import os
from lib.models.cttrack.cttrack import build_cttrack
from lib.test.utils.pre_processor import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box
from copy import deepcopy
from lib.utils.image import *
from lib.test.utils.vis_attn_maps import vis_attn_maps

class CTTrack(BaseTracker):
	def __init__(self, params, dataset_name):
		super(CTTrack, self).__init__(params)
		network = build_cttrack(params.cfg)
		missing_keys, unexpected_keys = network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
		print("load checkpoint: ", self.params.checkpoint)
		print("missing keys:", missing_keys)
		print("unexpected keys:", unexpected_keys)
		self.cfg = params.cfg
		self.network = network.cuda()
		self.network.eval()
		self.seq_id=None
		self.vis_attn=False
		self.preprocessor = Preprocessor_wo_mask()
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
		# self.z_dict1 = {}

		# Set the update interval
		DATASET_NAME = dataset_name.upper()
		if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
			self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
		else:
			self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
		print("Update interval is: ", self.update_intervals)

	def initialize(self, image, info: dict):
		# forward the template once
		z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
		                                            output_sz=self.params.template_size)

		if self.vis_attn:
			self.z_patch=z_patch_arr
			self.oz_patch=z_patch_arr

		template, template_amask = self.preprocessor.process(z_patch_arr, z_amask_arr)
		self.template = template
		self.online_template = deepcopy(template)
		self.template_amask = template_amask
		self.online_template_amask = deepcopy(template_amask)
		# print("template shape: {}".format(template.shape))
		# with torch.no_grad():
		#     self.z_dict1 = self.network.forward_backbone(template)
		# save states
		self.state = info['init_bbox']
		self.frame_id = 0
		if self.save_all_boxes:
			'''save all predicted boxes'''
			all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
			return {"all_boxes": all_boxes_save}

	def track(self, image, info: dict = None):
		H, W, _ = image.shape
		self.frame_id += 1
		if self.vis_attn and self.frame_id%200==0:
			image2=image.copy()
			
			cv2.rectangle(image2, pt1=(int(self.state[0]),int(self.state[1])), pt2=(int(self.state[0]+self.state[2]),int(self.state[1]+self.state[3])), color=(255, 0, 0),thickness=2)

			x_patch_arr2, resize_factor2, x_amask_arr2 = sample_target(image2, self.state, self.params.search_factor,
																	output_sz=self.params.search_size)  # (x1, y1, w, h)
		x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
		                                                        output_sz=self.params.search_size)  # (x1, y1, w, h)
		search, search_amask = self.preprocessor.process(x_patch_arr, x_amask_arr)

		if self.vis_attn and self.frame_id % 200 == 0:
			attn_weights = []
			hooks = []
			for i in range(len(self.network.backbone.blocks)):
				hooks.append(self.network.backbone.blocks[i].attn.attn_drop.register_forward_hook(
					lambda self, input, output: attn_weights.append(output)))

		with torch.no_grad():
			# # run the transformer
			out_dict, _ = self.network(template=self.template, online_template=self.online_template, search=search)
		# print("out_dict: {}".format(out_dict))

		if self.vis_attn and self.frame_id % 200 == 0:
			for hook in hooks:
				hook.remove()
			# attn0(t_ot) / 1(t_ot) / 2(t_ot_s)
			# shape: torch.Size([1, 12, 64, 128]), torch.Size([1, 12, 64, 128]), torch.Size([1, 12, 400, 528])
			# vis attn weights: online_template-to-template
			vis_attn_maps(attn_weights[1::4], q_w=8, k_w=8, skip_len=64, x1=self.oz_patch, x2=self.z_patch,
							x1_title='Online Template', x2_title='Template',
							save_path= 'vis_attn_weights/%04d/t2ot_vis/%04d' % (self.seq_id,self.frame_id))
			# vis attn weights: template-to-online_template
			vis_attn_maps(attn_weights[2::4], q_w=8, k_w=8, skip_len=0, x1=self.z_patch, x2=self.oz_patch,
							x1_title='Template', x2_title='Online Template',
							save_path='vis_attn_weights/%04d/ot2t_vis/%04d' % (self.seq_id,self.frame_id))
			# vis attn weights: template-to-search
			vis_attn_maps(attn_weights[3::4], q_w=20, k_w=8, skip_len=0, x1=self.z_patch, x2=x_patch_arr2,
							x1_title='Template', x2_title='Search',
							save_path='vis_attn_weights/%04d/s2t_vis/%04d' % (self.seq_id,self.frame_id))
			# vis attn weights: online_template-to-search
			vis_attn_maps(attn_weights[3::4], q_w=20, k_w=8, skip_len=64, x1=self.oz_patch, x2=x_patch_arr2,
							x1_title='Online Template', x2_title='Search',
							save_path='vis_attn_weights/%04d/s2ot_vis/%04d' % (self.seq_id,self.frame_id))
			# vis attn weights: search-to-search
			vis_attn_maps(attn_weights[3::4], q_w=20, k_w=20, skip_len=128, x1=x_patch_arr, x2=x_patch_arr2,
							x1_title='Search1', x2_title='Search2', idxs=[(160, 160)],
							save_path='vis_attn_weights/%04d/s2s_vis/%04d' % (self.seq_id,self.frame_id))
			print("save vis_attn of frame-{} done.".format(self.frame_id))

		pred_boxes = out_dict['pred_boxes'].view(-1, 4)
		# Baseline: Take the mean of all pred boxes as the final result
		pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
		# get the final box result
		self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

		# update template
		# for idx, update_i in enumerate(self.update_intervals):
		#     if self.frame_id % update_i == 0:
		#         z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
		#                                                     output_sz=self.params.template_size)  # (x1, y1, w, h)
		#         self.online_template = self.preprocessor.process(z_patch_arr)

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
	return CTTrack
