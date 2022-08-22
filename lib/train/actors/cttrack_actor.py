from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.image import *


class CTTrackActor(BaseActor):
	def __init__(self, net, objective, loss_weight, settings, run_score_head=False):
		super().__init__(net, objective)
		self.loss_weight = loss_weight
		self.settings = settings
		self.bs = self.settings.batchsize  # batch size
		self.run_score_head = run_score_head

	def __call__(self, data):
		"""
		args:
			data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
			template_images: (N_t, batch, 3, H, W)
			search_images: (N_s, batch, 3, H, W)
		returns:
			loss    - the training loss
			status  -  dict containing detailed losses
		"""
		# forward pass
		out_dict = self.forward_pass(data)

		# process the groundtruth
		gt_bboxes = data['search_bboxes']  # (Ns, batch, 4) (x1,y1,w,h)

		labels = None
		if 'pred_scores' in out_dict:
			try:
				labels = data['label'].view(-1)  # (batch, ) 0 or 1
			except:
				raise Exception("Please setting proper labels for score branch.")

		# compute losses
		loss, status = self.compute_losses(out_dict, labels = labels)

		return loss, status

	def forward_pass(self, data):
		out_dict, _ = self.net(data['template_images'][0], data['sequence_images'][0], data['search_images'][0],
		                       run_score_head=self.run_score_head)
		# out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
		return out_dict

	def compute_losses(self, pred_dict, return_status=True, labels=None):

		pred_scores = pred_dict['pred_scores'].view(-1)

		if torch.isnan(pred_scores).any():
			raise ValueError("Network outputs is NAN! Stop Training")

		score_loss = self.objective['score'](pred_scores, labels)
		loss = score_loss * self.loss_weight['score']

		if return_status:
			# status for log
			status = {"Loss/total": loss.item(),
			          "Loss/scores": score_loss.item()}

			return loss, status
		else:
			return loss
