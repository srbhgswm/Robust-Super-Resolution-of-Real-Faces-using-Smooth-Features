import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict as OD

class featureLoss(nn.Module):
	def __init__(self, num_layers):
		super(featureLoss, self).__init__();
		self.num_layers = num_layers;
		base_model = models.vgg19(pretrained=True).features.eval();
		for param in base_model.parameters():
			param.requires_grad = False;
		self.vgg19_ = nn.ModuleDict({\
			'layer1': nn.Sequential(*list(base_model)[:5]),\
			'layer2': nn.Sequential(*list(base_model)[5:10]),\
			'layer3': nn.Sequential(*list(base_model)[10:19]),\
			'layer4': nn.Sequential(*list(base_model)[19:28]),\
			'layer5': nn.Sequential(*list(base_model)[28:37])\
			});
		self.criterion = nn.MSELoss();

	def putThrough(self,input_tensor):
		for i in range(self.num_layers):
			input_tensor = self.vgg19_['layer{}'.format(i+1)](input_tensor);
		return input_tensor;

	def forward(self, inp, targ):
		for i in range(self.num_layers):
			self.vgg19_['layer_{}'.format(i+1)] = self.vgg19_['layer{}'.format(i+1)].to(inp.device);
		inp_feat = self.putThrough(inp);
		targ_feat = self.putThrough(targ);
		return self.criterion(inp_feat,targ_feat);

class featLoss(nn.Module):
	def __init__(self, num_layers):
		super(featLoss,self).__init__();
		self.num_layers = num_layers;
		base_model = models.vgg19(pretrained=True).features.eval();
		self.loss_criterion = nn.MSELoss();
		self.model = nn.ModuleDict({});
		i_start = 0; i_end = 1; l = 1;
		for layer in base_model.children():
			layer.requires_grad = False;
			if isinstance(layer,nn.Conv2d):
				self.model.update({'layer_%02d'%(l): nn.Sequential(*list(base_model)[i_start:i_end])})
				l += 1;
				i_start = i_end;
			i_end += 1;
			if isinstance(layer, nn.ReLU):
				layer.inplace = False;
			if (l >self.num_layers):
				break; 

	def computeFeat(self, input_tensor):
		for i in range(self.num_layers):
			input_tensor = self.model['layer_%02d'%(i+1)](input_tensor);
		return input_tensor;


	def forward(self, inp, targ):
		for i in range(self.num_layers):
			self.model['layer_%02d'%(i+1)] = self.model['layer_%02d'%(i+1)].to(inp.device);
		inp = self.computeFeat(inp);
		targ = self.computeFeat(targ).detach();
		loss = self.loss_criterion(inp, targ);
		return loss;

class featLoss2(nn.Module):
	def __init__(self, num_layers):
		super(featLoss2,self).__init__();
		self.num_layers = num_layers;
		base_model = models.vgg19(pretrained=True).features.eval();
		self.loss_criterion = nn.L1Loss();
		self.model = nn.ModuleDict({});
		i_start = 0; i_end = 1; l = 1;
		for layer in base_model.children():
			layer.requires_grad = False;
			if isinstance(layer,nn.Conv2d):
				self.model.update({'layer_%02d'%(l): nn.Sequential(*list(base_model)[i_start:i_end])})
				l += 1;
				i_start = i_end;
			i_end += 1;
			if isinstance(layer, nn.ReLU):
				layer.inplace = False;
			if (l >self.num_layers):
				break; 

	def computeFeat(self, input_tensor):
		for i in range(self.num_layers):
			input_tensor = self.model['layer_%02d'%(i+1)](input_tensor);
		return input_tensor;


	def forward(self, inp, targ):
		for i in range(self.num_layers):
			self.model['layer_%02d'%(i+1)] = self.model['layer_%02d'%(i+1)].to(inp.device);
		inp = self.computeFeat(inp);
		targ = self.computeFeat(targ).detach();
		loss = self.loss_criterion(inp, targ);
		return loss;
