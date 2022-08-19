# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : load_pretrain.py
# Copyright (c) Skye-Song. All Rights Reserved
def remove_prefix(state_dict, prefix):
	''' Old style model is stored with all names of parameters
	share common prefix 'module.' '''
	f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
	return {f(key): value for key, value in state_dict.items()}


def _get_prefix_dic(dict, prefix):
	f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
	return {f(key): value for key, value in dict.items() if key.startswith(prefix)}


def _add_prefix_dic(dict, prefix):
	f = lambda x: str(prefix) + str(x)
	return {f(key): value for key, value in dict.items()}
