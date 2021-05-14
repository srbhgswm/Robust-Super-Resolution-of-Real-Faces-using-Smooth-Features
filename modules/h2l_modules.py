import torch
import torch.nn as nn

class DiscRes(nn.Module):
	def __init__(self,nf):
		super(DiscRes,self).__init__();
		self.act = nn.LeakyReLU(negative_slope=0.2);
		self.l1 = nn.Conv2d(nf,nf,(3,3),padding=1,stride=1);
		self.l2 = nn.Conv2d(nf,nf,(3,3),padding=1,stride=1);

	def forward(self,input_tensor):
		temp = input_tensor;
		input_tensor = self.act(input_tensor);
		input_tensor = self.l1(input_tensor);
		input_tensor = self.act(input_tensor);
		input_tensor = self.l2(input_tensor);
		input_tensor = temp + input_tensor;
		return input_tensor;

class GenRes(nn.Module):
	def __init__(self,nf):
		super(GenRes,self).__init__();
		self.act = nn.PReLU();
		self.b1 = nn.BatchNorm2d(nf);
		self.b2 = nn.BatchNorm2d(nf);
		self.l1 = nn.Conv2d(nf,nf,(3,3),padding=1,stride=1);
		self.l2 = nn.Conv2d(nf,nf,(3,3),padding=1,stride=1);

	def forward(self,input_tensor):
		temp = input_tensor;
		input_tensor = self.b1(input_tensor);
		input_tensor = self.act(input_tensor);
		input_tensor = self.l1(input_tensor);
		input_tensor = self.b2(input_tensor);
		input_tensor = self.act(input_tensor);
		input_tensor = self.l2(input_tensor);
		input_tensor = temp + input_tensor;
		return input_tensor;

class Gen(nn.Module):
	def __init__(self,nf,latent=16):
		super(Gen,self).__init__();
		self.begin_latent = nn.Linear(latent,64);
		self.begin = nn.Sequential(nn.Conv2d(3+1,nf,(3,3),padding=1,stride=1),nn.PReLU());
		self.l1 = nn.Sequential(GenRes(nf),GenRes(nf),nn.AvgPool2d(kernel_size=2,stride=2)); #64 to 32
		self.l2 = nn.Sequential(GenRes(nf),GenRes(nf),nn.AvgPool2d(kernel_size=2,stride=2)); #32 to 16
		self.l3 = nn.Sequential(GenRes(nf),GenRes(nf),nn.AvgPool2d(kernel_size=2,stride=2)); #16 to 8
		self.l4 = nn.Sequential(GenRes(nf),GenRes(nf),nn.AvgPool2d(kernel_size=2,stride=2)); #8 to 4
		self.l5 = nn.Sequential(GenRes(nf),GenRes(nf),nn.PixelShuffle(2),nn.Conv2d(nf//4,nf,(1,1)),nn.PReLU()); #4 to 8
		self.l6 = nn.Sequential(GenRes(nf),GenRes(nf),nn.PixelShuffle(2),nn.Conv2d(nf//4,nf,(1,1)),nn.PReLU()); #8 to 16
		self.end = nn.Conv2d(nf,3,(3,3),padding=1,stride=1);

	def forward(self,input_tensor, latent_tensor):
		latent_tensor = self.begin_latent(latent_tensor).view(-1,1,1,64).expand(-1,1,64,64);
		input_tensor = self.begin(torch.cat((latent_tensor,input_tensor), dim=1));
		input_tensor = self.l1(input_tensor);
		input_tensor = self.l2(input_tensor);
		input_tensor = self.l3(input_tensor);
		input_tensor = self.l4(input_tensor);
		input_tensor = self.l5(input_tensor);
		input_tensor = self.l6(input_tensor);
		input_tensor = self.end(input_tensor);
		return input_tensor;

class Disc(nn.Module):
	def __init__(self,nf):
		super(Disc,self).__init__();
		self.begin = nn.Sequential(nn.Conv2d(3,nf,(3,3),padding=1,stride=1), nn.LeakyReLU(negative_slope=0.2));
		self.l1 = DiscRes(nf);
		self.l2 = DiscRes(nf);
		self.l3 = DiscRes(nf);
		self.l4 = DiscRes(nf);
		self.l5 = nn.Sequential(DiscRes(nf),nn.AvgPool2d(kernel_size=2,stride=2)); #16 to 8
		self.l6 = nn.Sequential(DiscRes(nf),nn.AvgPool2d(kernel_size=2,stride=2)); #8 to 4
		self.end = nn.Linear(1024,1);

	def forward(self,input_tensor):
		input_tensor = self.begin(input_tensor);
		input_tensor = self.l1(input_tensor);
		input_tensor = self.l2(input_tensor);
		input_tensor = self.l3(input_tensor);
		input_tensor = self.l4(input_tensor);
		input_tensor = self.l5(input_tensor);
		input_tensor = self.l6(input_tensor);
		input_tensor = input_tensor.view(input_tensor.shape[0],-1);
		input_tensor = self.end(input_tensor);
		return input_tensor;