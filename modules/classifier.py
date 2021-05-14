import torch
import torch.nn as nn
import torchvision.models as models

class mockLayer(nn.Module):
	def __init__(self):
		super(mockLayer,self).__init__();
	def forward(self,input_tensor):
		print(input_tensor.shape);
		return input_tensor;

#Importing the appropriate model
class Cls(nn.Module):
	def __init__(self):
		super(Cls,self).__init__();
		self.mu = nn.parameter.Parameter(torch.Tensor([0.485, 0.456, 0.406]), requires_grad=False);
		self.sigma = nn.parameter.Parameter(torch.Tensor([0.229, 0.224, 0.225]), requires_grad=False);
		self.net = models.resnet18(pretrained=True);
		for param in self.net.parameters():
			param.requires_grad = False;
		self.net.fc = nn.Sequential(\
			nn.Linear(in_features=2048, out_features=128, bias=True),\
			nn.Linear(in_features=128, out_features=1, bias=True)\
			);
	def forward(self,input_tensor):
		input_tensor = 0.5*input_tensor+0.5;
		for i in range(3):
			input_tensor = (input_tensor - self.mu[0]) / self.sigma[0]; 
		input_tensor = self.net(input_tensor.repeat(1,1,4,4));
		return input_tensor;
