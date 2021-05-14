import torch
import torch.nn as nn
from fastai.layers import Flatten

class RCABP(nn.Module):
	def __init__(self,inf,onf):
		super(RCABP,self).__init__();
		self.proc1 = nn.Sequential(nn.Conv2d(inf,onf,(3,3),padding=1,stride=1), nn.BatchNorm2d(onf),nn.PReLU(),\
			nn.Conv2d(onf,onf,(3,3),padding=1,stride=1), nn.BatchNorm2d(onf), nn.PReLU(),\
			nn.Conv2d(onf,onf,(3,3),padding=1,stride=1), nn.BatchNorm2d(onf), nn.PReLU());
		self.proc2 = nn.Sequential(nn.Conv2d(inf,onf,(3,3),padding=1,stride=1), nn.BatchNorm2d(onf), nn.PReLU(),\
			nn.Conv2d(onf,onf,(3,3),padding=1,stride=1), nn.BatchNorm2d(onf), nn.PReLU(),\
			nn.Conv2d(onf,onf,(3,3),padding=1,stride=1), nn.BatchNorm2d(onf), nn.PReLU());
		self.avg = nn.AdaptiveAvgPool2d((1,1));
		self.bottleneck = nn.Sequential(nn.Linear(onf,onf//4), nn.PReLU());
		self.unravel = nn.Sequential(nn.Linear(onf//4,onf),nn.Sigmoid());

	def forward(self,input_tensor):
		temp = self.proc1(input_tensor);
		input_tensor = self.proc2(input_tensor);
		temp2 = input_tensor;
		input_tensor = self.avg(input_tensor).view(input_tensor.shape[0],-1);
		input_tensor = self.bottleneck(input_tensor);
		input_tensor = self.unravel(input_tensor).view(input_tensor.shape[0],-1,1,1);
		input_tensor = input_tensor*temp2 + temp;
		return input_tensor;
		
class RCA(nn.Module):
	def __init__(self,nf):
		super(RCA,self).__init__();
		self.proc = nn.Sequential(nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU(),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU(),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.avg = nn.AdaptiveAvgPool2d((1,1));
		self.bottleneck = nn.Sequential(nn.Linear(nf,nf//4),nn.PReLU());
		self.unravel = nn.Sequential(nn.Linear(nf//4,nf),nn.Sigmoid());

	def forward(self,input_tensor):
		temp1 = input_tensor;
		input_tensor = self.proc(input_tensor);
		temp = input_tensor;
		input_tensor = self.avg(input_tensor);
		input_tensor = input_tensor.view(input_tensor.shape[0],-1);
		input_tensor = self.bottleneck(input_tensor);
		input_tensor = self.unravel(input_tensor).view(input_tensor.shape[0],-1,1,1);
		input_tensor = input_tensor*temp + temp1;
		return input_tensor;

class DenseBlock(nn.Module):
	def __init__(self,nf):
		super(DenseBlock,self).__init__();
		self.l1 = nn.Sequential(RCA(nf), RCABP(nf,nf));
		self.l2 = nn.Sequential(RCA(2*nf), RCABP(2*nf,nf));
		self.l3 = nn.Sequential(RCA(3*nf), RCABP(3*nf,nf));
		self.l4 = nn.Sequential(RCA(4*nf), RCABP(4*nf,nf));
		self.l5 = nn.Sequential(RCA(5*nf), RCABP(5*nf,nf));

	def forward(self,input_tensor):
		temp = input_tensor;
		input_tensor = torch.cat([input_tensor, self.l1(input_tensor)], dim=1);
		input_tensor = torch.cat([input_tensor, self.l2(input_tensor)], dim=1);
		input_tensor = torch.cat([input_tensor, self.l3(input_tensor)], dim=1);
		input_tensor = torch.cat([input_tensor, self.l4(input_tensor)], dim=1);
		input_tensor = self.l5(input_tensor);
		input_tensor = temp + 0.2*input_tensor;
		return input_tensor;


class ResBlock(nn.Module):
	def __init__(self,nf):
		super(ResBlock,self).__init__();
		self.l1 = nn.Sequential(nn.Conv2d(nf,2*nf,(3,3),stride=1,padding=1), nn.GroupNorm(4,2*nf), nn.PReLU());
		self.l2 = nn.Sequential(nn.Conv2d(2*nf,2*nf,(3,3),stride=1,padding=1), nn.GroupNorm(4,2*nf), nn.PReLU());
		self.l3 = nn.Sequential(nn.Conv2d(2*nf,nf,(3,3),stride=1,padding=1), nn.GroupNorm(4,nf), nn.PReLU());

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
		self.begin = nn.Sequential(nn.Conv2d(3,nf,(3,3),stride=1,padding=1), nn.PReLU());

		self.d1 = nn.Sequential(\
			RCA(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.d2 = nn.Sequential(\
			RCA(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.d3 = nn.Sequential(\
			RCA(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.d4 = nn.Sequential(\
			RCA(nf),\
			nn.Conv2d(nf,nf,(3,3),padding=1,stride=2), nn.BatchNorm2d(nf), nn.PReLU()\
			);
		self.end = nn.Sequential(nn.Conv2d(nf,nf,(1,1)), nn.PReLU());

	def forward(self,input_tensor):
		input_tensor = self.begin(input_tensor);
		input_tensor = self.d1(input_tensor);
		input_tensor = self.d2(input_tensor);
		input_tensor = self.d3(input_tensor);
		input_tensor = self.d4(input_tensor);
		input_tensor = self.end(input_tensor);
		return input_tensor;

class Dec(nn.Module):
	def __init__(self,nf):
		super(Dec,self).__init__();\
		self.up_b_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(nf,nf,(5,5),padding=2,stride=1), nn.BatchNorm2d(nf), nn.PReLU());
		self.l1 = DenseBlock(nf);
		self.up_1 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(nf//4,nf,(5,5),padding=2,stride=1), nn.BatchNorm2d(nf), nn.PReLU());
		self.up_2 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(nf//4,nf,(5,5),padding=2,stride=1), nn.BatchNorm2d(nf), nn.PReLU());
		self.up_b_2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(nf,nf,(5,5),padding=2,stride=1), nn.BatchNorm2d(nf), nn.PReLU());		
		self.l2 = DenseBlock(nf);
		self.up_3 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(nf//4,nf,(5,5),padding=2,stride=1), nn.BatchNorm2d(nf), nn.PReLU());
		self.up_4 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(nf//4,nf,(5,5),padding=2,stride=1), nn.BatchNorm2d(nf), nn.PReLU());

		self.end = nn.Conv2d(nf,3,(3,3),padding=1,stride=1);

	def forward(self,input_tensor):
		input_tensor = self.up_b_1(input_tensor);
		input_tensor = self.l1(input_tensor);
		input_tensor = self.up_1(input_tensor);
		input_tensor = self.up_2(input_tensor);
		input_tensor = self.up_b_2;
		input_tensor = self.l2(input_tensor);
		input_tensor = self.up_3(input_tensor);
		input_tensor = self.up_4(input_tensor);
		input_tensor = self.end(input_tensor);
		return input_tensor;

class Disc(nn.Module):
	def __init__(self,in_c=3,nf=64):
		super(Disc,self).__init__();
		self.begin = nn.Sequential(nn.Conv2d(in_c,nf,(3,3),stride=1,padding=1), nn.GroupNorm(4,nf), nn.PReLU());

		self.d_1 = nn.Sequential(\
			nn.Conv2d(nf,nf,(5,5),stride=2,padding=2),\
			nn.GroupNorm(4,nf),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.p_1 = nn.Sequential(\
			nn.Conv2d(nf,2*nf,(3,3),stride=1,padding=1),\
			nn.GroupNorm(4,2*nf),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.d_2 = nn.Sequential(\
			nn.Conv2d(2*nf,2*nf,(5,5),stride=2,padding=2),\
			nn.GroupNorm(4,2*nf),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.p_2 = nn.Sequential(\
			nn.Conv2d(2*nf,4*nf,(3,3),stride=1,padding=1),\
			nn.GroupNorm(4,4*nf),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.d_3 = nn.Sequential(\
			nn.Conv2d(4*nf,4*nf,(5,5),stride=2,padding=2),\
			nn.GroupNorm(4,4*nf),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.p_3 = nn.Sequential(\
			nn.Conv2d(4*nf,8*nf,(3,3),stride=1,padding=1),\
			nn.GroupNorm(4,8*nf),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.d_4 = nn.Sequential(\
			nn.Conv2d(8*nf,8*nf,(5,5),stride=2,padding=2),\
			nn.GroupNorm(4,8*nf),\
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
