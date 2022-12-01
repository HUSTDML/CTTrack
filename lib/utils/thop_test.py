import torch
from thop import profile,clever_format

def cal_model_size(model):
	template = torch.randn(1, 3, 128, 128).cuda()
	online_template = torch.randn(1, 3, 128, 128).cuda()
	search = torch.randn(1, 3, 320, 320).cuda()
	flops, params = profile(model, inputs=(template,online_template,search,True))
	print("flops:" + str(flops))
	print("params:" + str(params))
	flops, params = clever_format([flops, params], "%.3f")
	print("flops:" + str(flops))
	print("params:" + str(params))