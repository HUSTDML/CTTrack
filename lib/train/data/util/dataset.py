import random
import torch.utils.data
from lib.utils import TensorDict
from lib.utils.random import random_choice
import numpy as np
from lib.utils.image import *

def no_processing(data):
	return data

class SequenceDataset(torch.utils.data.Dataset):
	def __init__(self, datasets, p_datasets, samples_per_epoch,
	             num_search_frames, num_template_frames, num_sequence_frames, processing=no_processing,
	             train_cls=False, pos_prob=0.5):
		self.datasets = datasets
		self.train_cls = train_cls
		self.pos_prob = pos_prob

		if p_datasets is None:
			p_datasets = [len(d) for d in self.datasets]
		# Normalize
		p_total = sum(p_datasets)
		self.p_datasets = [x / p_total for x in p_datasets]

		self.samples_per_epoch = samples_per_epoch
		self.num_search_frames = num_search_frames
		self.num_template_frames = num_template_frames
		self.num_sequence_frames = num_sequence_frames
		self.processing = processing

	def __len__(self):
		return self.samples_per_epoch

	def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
	                        allow_invisible=False, force_invisible=False):
		""" Samples num_ids frames between min_id and max_id for which target is visible

		args:
			visible - 1d Tensor indicating whether target is visible for each frame
			num_ids - number of frames to be samples
			min_id - Minimum allowed frame number
			max_id - Maximum allowed frame number

		returns:
			list - List of sampled frame numbers. None if not sufficient visible frames could be found.
		"""

		if num_ids == 0:
			return []
		if min_id is None or min_id < 0:
			min_id = 0
		if max_id is None or max_id > len(visible):
			max_id = len(visible)
		# get valid ids
		if force_invisible:
			valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
		else:
			if allow_invisible:
				valid_ids = [i for i in range(min_id, max_id)]
			else:
				valid_ids = [i for i in range(min_id, max_id) if visible[i]]

		# No visible ids
		if len(valid_ids) == 0:
			return None

		return random_choice(valid_ids, k=num_ids, is_repeat=False)

	def __getitem__(self, index):
		if self.train_cls:
			return self.getitem_cls()
		else:
			return self.getitem()

	def getitem(self):
		"""
		returns:
			TensorDict - dict containing all the data blocks
		"""
		valid = False
		while not valid:
			# Select a dataset
			dataset = random.choices(self.datasets, self.p_datasets)[0]
			is_video_dataset = dataset.is_video_sequence()

			# sample a sequence
			seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
			seq_len = self.num_search_frames + self.num_template_frames + self.num_sequence_frames

			if is_video_dataset:
				# sample a visible frame
				frame_ids = self._sample_visible_ids(visible, num_ids=seq_len, allow_invisible=False)

				# change the frame_id sort randomly
				if random.random() < 0.5:
					frame_ids = sorted(frame_ids, reverse=False)
				else:
					frame_ids = sorted(frame_ids, reverse=True)
			else:
				frame_ids = [1] * seq_len

			if frame_ids is None:
				continue

			try:
				frames, anno, meta_obj = dataset.get_frames(seq_id, frame_ids, seq_info_dict)
				H, W, _ = frames[0].shape
				masks = anno['mask'] if 'mask' in anno else [torch.zeros((H, W))] * seq_len

				template_ids = [0]
				if self.num_template_frames > 1:
					template_ids += random_choice(list(range(1, self.num_template_frames + self.num_sequence_frames)),
					                              k=self.num_template_frames - 1, is_repeat=False)

				sequence_ids = random_choice(list(range(1, self.num_template_frames + self.num_sequence_frames)),
				                             k=self.num_sequence_frames, is_repeat=False)

				data = TensorDict({'template_images': [frames[i] for i in template_ids],
				                   'template_bboxes': [anno['bbox'][i] for i in template_ids],
				                   'template_masks': [masks[i] for i in template_ids],
				                   'sequence_images': [frames[i] for i in sequence_ids],
				                   'sequence_bboxes': [anno['bbox'][i] for i in sequence_ids],
				                   'sequence_masks': [masks[i] for i in sequence_ids],
				                   'search_images': [frames[-1]],
				                   'search_bboxes': [anno['bbox'][-1]],
				                   'search_masks': [masks[-1]],
				                   'dataset': dataset.get_name(),
				                   'test_class': meta_obj.get('object_class_name')})
				# make data augmentation
				data = self.processing(data)
				# check whether data is valid
				valid = data['valid']
			except:
				print("dataset load fail: " + dataset.get_name() + ", seq_id: " + str(seq_id)  + ", frame_ids: " + str(frame_ids) )
				valid = False

		return data

	def getitem_cls(self):
		valid = False
		while not valid:
			# Select a dataset
			dataset = random.choices(self.datasets, self.p_datasets)[0]
			is_video_dataset = dataset.is_video_sequence()

			# sample a sequence
			seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
			# sample a visible frame
			seq_len = self.num_sequence_frames + self.num_template_frames + self.num_search_frames

			if is_video_dataset:
				# sample a visible frame
				frame_ids = self._sample_visible_ids(visible, num_ids=seq_len, allow_invisible=False)
				# change the frame_id sort randomly
				if random.random() < 0.5:
					frame_ids = sorted(frame_ids, reverse=False)
				else:
					frame_ids = sorted(frame_ids, reverse=True)
			else:
				frame_ids = [1] * seq_len
			if frame_ids is None:
				continue

			try:
				frames, anno, meta_obj = dataset.get_frames(seq_id, frame_ids, seq_info_dict)
				H, W, _ = frames[0].shape
				masks = anno['mask'] if 'mask' in anno else [torch.zeros((H, W))] * seq_len

				template_ids = [0]

				if self.num_template_frames > 1:
					template_ids += random_choice(list(range(1, self.num_template_frames + self.num_sequence_frames)),
					                              k=self.num_template_frames - 1, is_repeat=False)

				sequence_ids = random_choice(list(range(1, self.num_template_frames + self.num_sequence_frames)),
				                             k=self.num_sequence_frames, is_repeat=False)

				# invisible search image
				if random.random() < self.pos_prob:
					label = torch.ones(1, )
					search_images = [frames[-1]]
					search_bboxes = [anno['bbox'][-1]]
					search_masks = [masks[-1]]
				else:
					label = torch.zeros(1, )

					search_frame_ids = self._sample_visible_ids(visible, num_ids=1, force_invisible=True)
					if search_frame_ids is None or is_video_dataset:
						search_images, search_anno, meta_obj_test = self.get_one_search()
					else:
						search_images, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids,
						                                                               seq_info_dict)

					H, W, _ = search_images[0].shape
					search_bboxes = [self.get_center_box(H, W)]
					search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros((H, W))]

				data = TensorDict({'template_images': [frames[i] for i in template_ids],
				                   'template_bboxes': [anno['bbox'][i] for i in template_ids],
				                   'template_masks': [masks[i] for i in template_ids],
				                   'sequence_images': [frames[i] for i in sequence_ids],
				                   'sequence_bboxes': [anno['bbox'][i] for i in sequence_ids],
				                   'sequence_masks': [masks[i] for i in sequence_ids],
				                   'search_images': search_images,
				                   'search_bboxes': search_bboxes,
				                   'search_masks': search_masks,
				                   'dataset': dataset.get_name(),
				                   'test_class': meta_obj.get('object_class_name')})
				# make data augmentation
				data = self.processing(data)
				data["label"] = label
				# check whether data is valid
				valid = data['valid']
			except:
				print("dataset load fail: " + dataset.get_name() + ", seq_id: " + str(seq_id)  + ", frame_ids: " + str(frame_ids) )
				valid = False
		return data

	def get_one_search(self):
		# Select a dataset
		dataset = random.choices(self.datasets, self.p_datasets)[0]

		is_video_dataset = dataset.is_video_sequence()
		# sample a sequence
		seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
		# sample a frame
		if is_video_dataset:
			search_frame_ids = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
		else:
			search_frame_ids = [1]
		# get the image, bounding box and other info
		search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

		return search_frames, search_anno, meta_obj_test

	def sample_seq_from_dataset(self, dataset, is_video_dataset):

		# Sample a sequence with enough visible frames
		enough_visible_frames = False
		while not enough_visible_frames:
			# Sample a sequence
			seq_id = random.randint(0, dataset.get_num_sequences() - 1)
			# seq_id = 1

			# Sample frames
			seq_info_dict = dataset.get_sequence_info(seq_id)
			visible = seq_info_dict['visible']

			enough_visible_frames = visible.type(torch.int64).sum().item() > \
			                        2 * (self.num_search_frames + self.num_template_frames + self.num_sequence_frames) \
			                        and len(visible) >= 30

			enough_visible_frames = enough_visible_frames or not is_video_dataset
		return seq_id, visible, seq_info_dict

	def get_center_box(self, H, W, ratio=1 / 8):
		cx, cy, w, h = W / 2, H / 2, W * ratio, H * ratio
		return torch.tensor([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
