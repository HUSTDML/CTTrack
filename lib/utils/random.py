# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : random.py
# Copyright (c) Skye-Song. All Rights Reserved

import random


def random_choice(l: list, k: int, is_repeat=True):
	if is_repeat:
		return random.choices(l, k=k)
	else:
		r = []
		for i in range(k):
			temp = random.choice(l)
			r.append(temp)
			l.remove(temp)
		return r