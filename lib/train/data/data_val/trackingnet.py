# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : trackingnet.py
# Copyright (c) Skye-Song. All Rights Reserved
import os
import numpy as np
from shutil import copyfile

# data_path = "/home/hust/Tracking/TrackingNet/"


from_path = "/group_homes/public_cluster/home/ud201980980/tracking/TrackingNet/"
to_path = "/group_homes/GxDVisionTasks/home/share/Tracking/TrackingNet/"
# to_path = "/home/ud201980980/tracking/TrackingNet/"

def load_text(path, delimiter, dtype):
	if isinstance(delimiter, (tuple, list)):
		for d in delimiter:
			try:
				ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
				return ground_truth_rect
			except:
				pass

		raise Exception('Could not read file {}'.format(path))
	else:
		ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
		return ground_truth_rect


class Sequence():
	def __init__(self, base_path, name=None, anno_name=None):
		self.base_path = base_path

		if name is not None:
			self.name = name
		else:
			self.name = anno_name[:-4]

		if anno_name is not None:
			self.anno_name = anno_name
		else:
			self.anno_name = name + ".txt"

		self.img_dir_path = os.path.join(base_path, "zips", self.name)
		self.anno_file_path = os.path.join(base_path, "anno", self.anno_name)

		self.cal_img_num()
		self.cal_anno_num()

	def cal_img_num(self):
		try:
			img_sets = os.listdir(self.img_dir_path)
		except:
			img_sets = []
		self.img_names = sorted(img_sets)
		self.img_num = len(self.img_names)

	def cal_anno_num(self):
		ground_truth_rect = load_text(self.anno_file_path, delimiter=',', dtype=np.float64)
		self.anno_num = ground_truth_rect.shape[0]

	def get_missing_img_names(self):
		missing_img_name = []

		if self.img_num == self.anno_num:
			return missing_img_name

		for i in range(self.anno_num):
			i_name = "{}.jpg".format(i)
			if i_name not in self.img_names:
				missing_img_name.append(i_name)
		return missing_img_name


def checkdata(from_path, to_path, check=True):
	# sets = ['TEST']
	data_path = to_path
	sets = ['TRAIN_{}'.format(i) for i in reversed(range(12))]
	for set in sets:
		print("val " + set)
		base_path = os.path.join(data_path, set)
		anno_path = os.path.join(base_path, "anno")
		sequences = os.listdir(anno_path)
		seq_set = []
		for anno_name in sequences:
			seq_class = Sequence(base_path, anno_name=anno_name)
			missing_imgs = seq_class.get_missing_img_names()
			if len(missing_imgs) > 0:
				print(os.path.join(set, "zips", seq_class.name) + "---all:"+ str(seq_class.anno_num)  +"---missing:" + str(len(missing_imgs)))

				if check:
					continue

				for miss_img in missing_imgs:
					from_img = os.path.join(from_path, set, "zips", seq_class.name, miss_img)
					to_img = os.path.join(to_path, set, "zips", seq_class.name, miss_img)
					copyfile(from_img, to_img)

			seq_set.append(seq_class)
		print(set + " done!")
		print("\n")


if __name__ == '__main__':
	checkdata(from_path, to_path, check=True)
