import torch
from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy

class CTTrackTrainActor(BaseActor):
	def __init__(self, net, objective, loss_weight, settings):
		super().__init__(net, objective)
		self.loss_weight = loss_weight
		self.settings = settings
		self.bs = self.settings.batchsize  # batch size

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
		renew_loss, out_dict = self.forward_pass(data)

		# process the groundtruth
		gt_bboxes = data['search_bboxes']  # (Ns, batch, 4) (x1,y1,w,h)
		# target_imgs = data['template_images']
		# compute losses
		loss, status = self.compute_losses(renew_loss, out_dict, gt_bboxes[0])

		return loss, status

	def forward_pass(self, data):
		renew_loss, out_dict, _ = self.net(data['template_images'][0],
		                                   data['sequence_images'][0],
		                                   data['search_images'][0],
		                                   data['target_in_search_images'][0],
		                                   data['search_bboxes'][0],
		                                   t2t=self.settings.loss_t2t,
		                                   t2s=self.settings.loss_t2s,
		                                   s2t=self.settings.loss_s2t,
		                                   s2s=self.settings.loss_s2s
		                                   )
		# out_dict: (B, N, C)
		return renew_loss, out_dict

	def compute_losses(self, renew_loss, pred_dict, gt_bbox, return_status=True):
		# Get boxes
		pred_boxes = pred_dict['pred_boxes']
		if torch.isnan(pred_boxes).any():
			raise ValueError("Network outputs is NAN! Stop Training")
		num_queries = pred_boxes.size(1)
		pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
		gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
		                                                                                                   max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

		# compute giou and iou
		try:
			giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
		except:
			giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

		# compute l1 loss
		l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

		# weighted sum
		loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + \
		       self.loss_weight['renew'] * renew_loss

		# loss = renew_loss
		if return_status:
			# status for log
			mean_iou = iou.detach().mean()
			status = {"Loss/total": loss.item(),
			          "Loss/record": loss.item() - renew_loss.item(),
			          "Loss/giou": giou_loss.item(),
			          "Loss/l1": l1_loss.item(),
			          "Loss/renew": renew_loss.item(),
			          "IoU": mean_iou.item()}
			return loss, status
		else:
			return loss
