# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : image.py
# Copyright (c) Skye-Song. All Rights Reserved

import os.path
import numpy as np
import cv2
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import math


def load_image(img_file: str) -> np.array:
	"""Image loader used by data module (e.g. image sampler)

	Parameters
	----------
	img_file: str
		path to image file
	Returns
	-------
	np.array
		loaded image

	Raises
	------
	FileExistsError
		invalid image file
	RuntimeError
		unloadable image file
	"""
	if not os.path.isfile(img_file):
		logger.info("Image file %s does not exist." % img_file)
	# read with OpenCV
	img = cv2.imread(img_file, cv2.IMREAD_COLOR)
	if img is None:
		logger.info("Fail to load Image file %s" % img_file)
	return img


def draw_image(img, box=None, norm_image=False, color='r'):
	img = uniform(img, norm_img=norm_image)
	# img = np.array(img, dtype=int)
	fig, ax = plt.subplots(1)
	ax.imshow(img)
	if box is not None:
		rect = patches.Rectangle(box[:2], box[2], box[3], linewidth=2, fill=False, edgecolor=color)
		ax.add_patch(rect)
	plt.show()


def draw_tensor(a: torch.Tensor):
	a_np = a.squeeze().cpu().clone().detach().numpy()
	if a_np.ndim == 3:
		a_np = np.transpose(a_np, (1, 2, 0))
	fig, ax = plt.subplots(1)
	ax.imshow(a_np)
	plt.show()


def draw_feat(a: torch.Tensor, fix_max_min=True):
	# a = (H,W,channel)
	one_channel = False
	if len(a.squeeze().size()) == 2:
		H, W = a.squeeze().size()
		C = 1
		one_channel = True
	elif len(a.squeeze().size()) == 3:
		H, W, C = a.squeeze().size()
	else:
		return

	a_np = a.squeeze().cpu().clone().detach().numpy()

	if fix_max_min:
		max = 1.5
		min = -1.5
	else:
		max = np.max(a_np)
		min = np.min(a_np)

	a_np = (np.minimum(np.maximum(a_np, min), max) - min) / (max - min) * 255.
	a_np = a_np.astype(int)

	h_num = int(np.ceil(math.sqrt(C)))

	image = np.zeros([H * h_num + h_num - 1, W * h_num + h_num - 1])

	flag = True
	for i in range(h_num):
		for j in range(h_num):
			index = i * h_num + j
			if index >= C:
				flag = False
				break
			if one_channel:
				image[i * (H + 1):i * (H + 1) + H, j * (W + 1):j * (W + 1) + W] = a_np
			else:
				image[i * (H + 1):i * (H + 1) + H, j * (W + 1):j * (W + 1) + W] = a_np[:, :, index]
		if not flag:
			break

	draw_image(image)


def uniform(images, norm_img=False):
	if isinstance(images, torch.Tensor):
		images = images.detach().cpu().numpy()

	std = np.array([0.229, 0.224, 0.225])
	mean = np.array([0.485, 0.456, 0.406])

	if len(images.shape) == 3:
		std, mean = std.reshape(1, 1, -1), mean.reshape(1, 1, -1)
		if images.shape[0] == 3:
			images = images.transpose(1, 2, 0)
	elif len(images.shape) == 4:
		std, mean = std.reshape(1, 1, 1, -1), mean.reshape(1, 1, 1, -1)
		if images.shape[1] == 3:
			images = images.transpose(0, 2, 3, 1)
	else:
		return images

	if norm_img:
		images = images * std + mean

	return images


def draw_seq_image(imgs, last_img=None, norm_img=False, crop_boxes=None, draw_boxes=None, color='r'):
	# 格式为 [seq_len, h, w, 3]
	imgs = uniform(imgs, norm_img=norm_img)
	temp_imgs = []
	if crop_boxes is not None:
		for i in range(len(imgs)):
			temp_imgs.append(crop_image(imgs[i], crop_boxes[i]))
		imgs = np.array(temp_imgs)

	if len(imgs.shape) == 4:
		[seq_len, h, w, _] = imgs.shape
	else:
		[seq_len, h, w] = imgs.shape
		imgs = imgs[:, :, :, np.newaxis].repeat(3, -1)

	canvas = np.ones([h, (w + 1) * seq_len, 3])

	if last_img is not None:
		last_img = uniform(last_img, norm_img=norm_img)
		canvas = np.ones([h, (w + 1) * (seq_len + 1), 3])
		canvas[:, seq_len * (w + 1):seq_len * (w + 1) + w, :] = last_img

	for i in range(seq_len):
		canvas[:, i * (w + 1):i * (w + 1) + w, :] = imgs[i]

	fig, ax = plt.subplots(1)
	ax.imshow(canvas)
	# if boxes is not None:
	# 	rect = patches.Rectangle(box[:2], box[2], box[3], linewidth=2, fill=False, edgecolor=color)
	# 	ax.add_patch(rect)
	plt.show()


def crop_image(im, target_bb, search_area_factor=None, output_sz=224):
	""" Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

	args:
		im - cv image
		target_bb - target box [x, y, w, h]
		search_area_factor - Ratio of crop size to target size
		output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

	returns:
		cv image - extracted crop
		float - the factor by which the crop has been resized to make the crop size equal output_size
	"""
	if not isinstance(target_bb, list):
		x, y, w, h = target_bb.tolist()
	else:
		x, y, w, h = target_bb

	if h == 0 or w == 0:
		raise Exception('wrong bounding box.')
	if search_area_factor is None:
		search_area_factor = min(max(math.sqrt(h / w), math.sqrt(w / h)), 2.0)

	# Crop image
	crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

	if crop_sz < 1:
		raise Exception('Too small bounding box.')

	x1 = round(x + 0.5 * w - crop_sz * 0.5)
	x2 = x1 + crop_sz

	y1 = round(y + 0.5 * h - crop_sz * 0.5)
	y2 = y1 + crop_sz

	x1_pad = max(0, -x1)
	x2_pad = max(x2 - im.shape[1] + 1, 0)

	y1_pad = max(0, -y1)
	y2_pad = max(y2 - im.shape[0] + 1, 0)

	# Crop target
	im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
	im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)

	if output_sz is not None:
		im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
		return im_crop_padded
	else:
		return im_crop_padded
