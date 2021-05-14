import torch
import torch.nn as nn
from pytorch_revgrad import RevGrad

class Critic_class(nn.Module):
	def __init__(self, embed_dim, transformed_sizes):
		super(Critic_class, self).__init__();
		transformed_sizes = [embed_dim] + transformed_sizes;
		self.fc = nn.Sequential(*[\
			nn.Sequential(nn.Linear(transformed_sizes[i], transformed_sizes[i+1]), nn.Dropout(0.2), nn.PReLU())\
			for i in range(len(transformed_sizes)-2)]);
		self.final = nn.Sequential(nn.Linear(transformed_sizes[-2],1));

	def forward(self, input_tensor):
		input_tensor = input_tensor;
		input_tensor = self.fc(input_tensor);
		input_tensor = self.final(input_tensor);
		return input_tensor;

class Critic_class_2(nn.Module):
	def __init__(self, nf):
		super(Critic_class_2, self).__init__();
		self.begin = nn.Sequential(nn.Conv2d(nf,nf//2,(5,5),padding=2,stride=2),\
		nn.BatchNorm2d(nf//2),\
		nn.LeakyReLU(),
		nn.Conv2d(nf//2,nf//4,(5,5),padding=2,stride=2),\
		nn.BatchNorm2d(nf//4),\
		nn.LeakyReLU());
		self.fc = nn.Sequential(nn.Linear(64,128), nn.BatchNorm1d(128), nn.LeakyReLU(),\
			nn.Linear(128,1)\
			)
		
	def forward(self, input_tensor):
		input_tensor = self.begin(input_tensor);
		input_tensor = input_tensor.view(input_tensor.shape[0],-1);
		input_tensor = self.fc(input_tensor);
		return input_tensor;


