import torch
import torch.nn as nn
from fastai.layers import Flatten

class ResBlock(nn.Module):
	def __init__(self,nf):
		super(ResBlock,self).__init__();
		self.l1 = nn.Sequential(nn.Conv2d(nf,2*nf,(3,3),stride=1,padding=1), nn.BatchNorm2d(2*nf), nn.PReLU());
		self.l2 = nn.Sequential(nn.Conv2d(2*nf,2*nf,(3,3),stride=1,padding=1), nn.BatchNorm2d(2*nf), nn.PReLU());
		self.l3 = nn.Sequential(nn.Conv2d(2*nf,nf,(3,3),stride=1,padding=1), nn.BatchNorm2d(nf), nn.PReLU());

	def forward(self,input_tensor):
		temp = input_tensor;
		input_tensor = self.l1(input_tensor);
		input_tensor = self.l2(input_tensor);
		input_tensor = self.l3(input_tensor);
		input_tensor = temp + 0.2*input_tensor;
		return input_tensor;

class Enc(nn.Module):
	def __init__(self,nf):
		super(Enc,self).__init__();
		self.begin = nn.Sequential(nn.Conv2d(3,nf,(3,3),stride=1,padding=1), nn.BatchNorm2d(nf), nn.PReLU());

		self.d1 = nn.Sequential(\
			ResBlock(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.d2 = nn.Sequential(\
			ResBlock(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.d3 = nn.Sequential(\
			ResBlock(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.d4 = nn.Sequential(\
			ResBlock(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.u1 = nn.Sequential(\
			ResBlock(nf),\
			nn.Upsample(scale_factor=2,mode='nearest'),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.u2 = nn.Sequential(\
			ResBlock(nf),\
			nn.Upsample(scale_factor=2,mode='nearest'),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.u3 = nn.Sequential(\
			ResBlock(nf),\
			nn.Upsample(scale_factor=2,mode='nearest'),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);

	def forward(self,input_tensor):
		input_tensor = self.begin(input_tensor);
		input_tensor = self.d1(input_tensor);
		temp1 = input_tensor;
		input_tensor = self.d2(input_tensor);
		temp2 = input_tensor;
		input_tensor = self.d3(input_tensor);
		temp3 = input_tensor;
		input_tensor = self.d4(input_tensor);
		input_tensor = self.u1(input_tensor);
		input_tensor = temp3 + 0.2*input_tensor;
		input_tensor = self.u2(input_tensor);
		input_tensor = temp2 + 0.2*input_tensor;
		input_tensor = self.u3(input_tensor);
		input_tensor = temp1 + 0.2*input_tensor;
		return input_tensor;

class Dec(nn.Module):
	def __init__(self,nf):
		super(Dec,self).__init__();\

		self.d1 = nn.Sequential(\
			ResBlock(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.d2 = nn.Sequential(\
			ResBlock(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.d3 = nn.Sequential(\
			ResBlock(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.u1 = nn.Sequential(\
			ResBlock(nf),\
			nn.Upsample(scale_factor=2,mode='nearest'),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.u2 = nn.Sequential(\
			ResBlock(nf),\
			nn.Upsample(scale_factor=2,mode='nearest'),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.u3 = nn.Sequential(\
			ResBlock(nf),\
			nn.Upsample(scale_factor=2,mode='nearest'),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.u4 = nn.Sequential(\
			ResBlock(nf),\
			nn.Upsample(scale_factor=2,mode='nearest'),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.end = nn.Conv2d(nf,3,(3,3),padding=1,stride=1);

	def forward(self,input_tensor):
		temp1 = input_tensor;
		input_tensor = self.d1(input_tensor);
		temp2 = input_tensor;
		input_tensor = self.d2(input_tensor);
		temp3 = input_tensor;
		input_tensor = self.d3(input_tensor);
		input_tensor = self.u1(input_tensor);
		input_tensor = temp3 + 0.2*input_tensor;
		input_tensor = self.u2(input_tensor);
		input_tensor = temp2 + 0.2*input_tensor;
		input_tensor = self.u3(input_tensor);
		input_tensor = temp1 + 0.2*input_tensor;
		input_tensor = self.u4(input_tensor);
		input_tensor = self.end(input_tensor);
		return input_tensor;

class Autoencoder(nn.Module):
	def __init__(self,nf):
		super(Autoencoder,self).__init__();
		self.enc = Enc(nf);
		self.dec = Dec(nf);

	def forward(self,input_tensor):
		outputs = {};
		input_tensor = self.enc(input_tensor);
		outputs['feat'] = input_tensor;
		input_tensor = self.dec(input_tensor);
		outputs['dec_out'] = input_tensor;
		return outputs;

class Gen(nn.Module):
	def __init__(self,nf):
		super(Gen,self).__init__();\

		self.d1 = nn.Sequential(\
			ResBlock(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.d2 = nn.Sequential(\
			ResBlock(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.d3 = nn.Sequential(\
			ResBlock(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.u1 = nn.Sequential(\
			ResBlock(nf),\
			nn.Upsample(scale_factor=2,mode='nearest'),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.u2 = nn.Sequential(\
			ResBlock(nf),\
			nn.Upsample(scale_factor=2,mode='nearest'),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.u3 = nn.Sequential(\
			ResBlock(nf),\
			nn.Upsample(scale_factor=2,mode='nearest'),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.u4 = nn.Sequential(\
			ResBlock(nf),\
			nn.Upsample(scale_factor=2,mode='nearest'),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.u5 = nn.Sequential(\
			ResBlock(nf),\
			nn.Upsample(scale_factor=2,mode='nearest'),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.u6 = nn.Sequential(\
			ResBlock(nf),\
			nn.Upsample(scale_factor=2,mode='nearest'),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.end = nn.Conv2d(nf,3,(3,3),padding=1,stride=1);

	def forward(self,input_tensor):
		temp1 = input_tensor;
		input_tensor = self.d1(input_tensor);
		temp2 = input_tensor;
		input_tensor = self.d2(input_tensor);
		temp3 = input_tensor;
		input_tensor = self.d3(input_tensor);
		input_tensor = self.u1(input_tensor);
		input_tensor = temp3 + 0.2*input_tensor;
		input_tensor = self.u2(input_tensor);
		input_tensor = temp2 + 0.2*input_tensor;
		input_tensor = self.u3(input_tensor);
		input_tensor = temp1 + 0.2*input_tensor;
		input_tensor = self.u4(input_tensor);
		input_tensor = self.u5(input_tensor);
		input_tensor = self.u6(input_tensor);
		input_tensor = self.end(input_tensor);
		return input_tensor;

class Disc(nn.Module):
	def __init__(self,in_c=3,nf=64):
		super(Disc,self).__init__();
		self.begin = nn.Sequential(nn.Conv2d(in_c,nf,(3,3),stride=1,padding=1), nn.BatchNorm2d(nf), nn.PReLU());

		self.d_1 = nn.Sequential(\
			nn.Conv2d(nf,nf,(5,5),stride=2,padding=2),\
			nn.BatchNorm2d(nf),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.p_1 = nn.Sequential(\
			nn.Conv2d(nf,2*nf,(3,3),stride=1,padding=1),\
			nn.BatchNorm2d(2*nf),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.d_2 = nn.Sequential(\
			nn.Conv2d(2*nf,2*nf,(5,5),stride=2,padding=2),\
			nn.BatchNorm2d(2*nf),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.p_2 = nn.Sequential(\
			nn.Conv2d(2*nf,4*nf,(3,3),stride=1,padding=1),\
			nn.BatchNorm2d(4*nf),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.d_3 = nn.Sequential(\
			nn.Conv2d(4*nf,4*nf,(5,5),stride=2,padding=2),\
			nn.BatchNorm2d(4*nf),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.p_3 = nn.Sequential(\
			nn.Conv2d(4*nf,8*nf,(3,3),stride=1,padding=1),\
			nn.BatchNorm2d(8*nf),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.d_4 = nn.Sequential(\
			nn.Conv2d(8*nf,8*nf,(5,5),stride=2,padding=2),\
			nn.BatchNorm2d(8*nf),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.end = nn.Sequential(\
			nn.AdaptiveAvgPool1d(1024),\
			nn.Linear(1024,1)\
			);

	def forward(self,input_tensor):
		input_tensor = self.begin(input_tensor);
		input_tensor = self.d_1(input_tensor);
		input_tensor = self.p_1(input_tensor);
		input_tensor = self.d_2(input_tensor);
		input_tensor = self.p_2(input_tensor);
		input_tensor = self.d_3(input_tensor);
		input_tensor = self.p_3(input_tensor);
		input_tensor = self.d_4(input_tensor);
		input_tensor = input_tensor.view(input_tensor.shape[0],1,-1);
		input_tensor = self.end(input_tensor);
		return input_tensor;
