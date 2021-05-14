import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class BPModule(nn.Module):
	def __init__(self,nf):
		super(BPModule,self).__init__();
		self.l1 = nn.Sequential(nn.Conv2d(nf,nf,(3,3),padding=1,stride=1), nn.LocalResponseNorm(2), nn.PReLU());
		self.l2 = nn.Sequential(nn.Conv2d(nf,nf//4,(3,3),padding=1,stride=1), nn.LocalResponseNorm(2), nn.PReLU());
		self.l3 = nn.Sequential(nn.Conv2d(nf//4,nf,(3,3),padding=1,stride=1), nn.LocalResponseNorm(2), nn.PReLU());

	def forward(self,input_tensor):
		temp = input_tensor;
		input_tensor = self.l1(input_tensor);
		input_tensor = self.l2(input_tensor);
		input_tensor = self.l3(input_tensor);
		input_tensor = temp + 0.2*input_tensor;
		return input_tensor;

class H2L(nn.Module):
	def __init__(self, nf=64, latent_dim=16):
		super(H2L,self).__init__();
		self.latent_dim = latent_dim;
		self.begin = nn.Sequential(nn.Conv2d(3+latent_dim,nf,(1,1)), nn.PReLU());

		self.l1 = nn.Sequential(BPModule(nf),\
			nn.Conv2d(nf,2*nf,(5,5),stride=2,padding=2), nn.LocalResponseNorm(2), nn.PReLU());
		self.l2 = nn.Sequential(BPModule(2*nf),\
			nn.Conv2d(2*nf,4*nf,(5,5),stride=2,padding=2), nn.LocalResponseNorm(2), nn.PReLU());
		self.l3 = nn.Sequential(BPModule(4*nf),\
			nn.Conv2d(4*nf,8*nf,(5,5),stride=2,padding=2), nn.LocalResponseNorm(2), nn.PReLU());
		self.u1 = nn.Sequential(BPModule(8*nf),\
			nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(8*nf,4*nf,(5,5),stride=1,padding=2), nn.LocalResponseNorm(2), nn.PReLU());
		self.u2 = nn.Sequential(BPModule(4*nf),\
			nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(4*nf,2*nf,(5,5),stride=1,padding=2), nn.LocalResponseNorm(2), nn.PReLU());
		self.u3 = nn.Sequential(BPModule(2*nf),\
			nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(2*nf,nf,(5,5),stride=1,padding=2), nn.LocalResponseNorm(2), nn.PReLU());

		self.end = nn.Sequential(nn.Conv2d(nf,3,(1,1)));

	def forward(self,input_tensor,latent):
		latent = latent.expand(*latent.shape[:2],*input_tensor.shape[2:]);
		input_tensor = torch.cat((input_tensor,latent), dim=1);
		input_tensor = self.begin(input_tensor);
		
		temp1 = input_tensor;
		input_tensor = self.l1(input_tensor);
		temp2 = input_tensor;
		input_tensor = self.l2(input_tensor);
		temp3 = input_tensor;
		input_tensor = self.l3(input_tensor);
		input_tensor = self.u1(input_tensor);
		input_tensor = temp3 + 0.2*input_tensor;
		input_tensor = self.u2(input_tensor);
		input_tensor = temp2 + 0.2*input_tensor;
		input_tensor = self.u3(input_tensor);
		input_tensor = temp1 + 0.2*input_tensor;

		input_tensor = self.end(input_tensor);
		return input_tensor;

class H2L2(nn.Module):
	def __init__(self, nf=64, latent_dim=16):
		super(H2L2,self).__init__();
		self.latent_dim = latent_dim;
		self.begin_latent = nn.Sequential(nn.Linear(latent_dim,latent_dim**2), nn.PReLU());
		self.begin = nn.Sequential(nn.Conv2d(3+1,nf,(1,1)), nn.PReLU());

		self.l1 = nn.Sequential(BPModule(nf),\
			nn.Conv2d(nf,2*nf,(5,5),stride=2,padding=2), nn.BatchNorm2d(2*nf), nn.PReLU());
		self.l2 = nn.Sequential(BPModule(2*nf),\
			nn.Conv2d(2*nf,4*nf,(5,5),stride=2,padding=2), nn.BatchNorm2d(4*nf), nn.PReLU());
		self.l3 = nn.Sequential(BPModule(4*nf),\
			nn.Conv2d(4*nf,8*nf,(5,5),stride=2,padding=2), nn.BatchNorm2d(8*nf), nn.PReLU());
		self.u1 = nn.Sequential(BPModule(8*nf),\
			nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(8*nf,4*nf,(5,5),stride=1,padding=2), nn.BatchNorm2d(4*nf), nn.PReLU());
		self.u2 = nn.Sequential(BPModule(4*nf),\
			nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(4*nf,2*nf,(5,5),stride=1,padding=2), nn.BatchNorm2d(2*nf), nn.PReLU());
		self.u3 = nn.Sequential(BPModule(2*nf),\
			nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(2*nf,nf,(5,5),stride=1,padding=2), nn.BatchNorm2d(nf), nn.PReLU());

		self.end = nn.Sequential(nn.Conv2d(nf,3,(1,1)), nn.Tanh());

	def forward(self,input_tensor,latent):
		latent = latent.view(latent.shape[0],-1);
		latent = self.begin_latent(latent).view(latent.shape[0],1,*input_tensor.shape[2:]);
		input_tensor = torch.cat((input_tensor,latent), dim=1);
		input_tensor = self.begin(input_tensor);
		
		temp1 = input_tensor;
		input_tensor = self.l1(input_tensor);
		temp2 = input_tensor;
		input_tensor = self.l2(input_tensor);
		temp3 = input_tensor;
		input_tensor = self.l3(input_tensor);
		input_tensor = self.u1(input_tensor);
		input_tensor = temp3 + 0.2*input_tensor;
		input_tensor = self.u2(input_tensor);
		input_tensor = temp2 + 0.2*input_tensor;
		input_tensor = self.u3(input_tensor);
		input_tensor = temp1 + 0.2*input_tensor;

		input_tensor = self.end(input_tensor);
		return input_tensor;


class Disc(nn.Module):
	def __init__(self,in_c=3,nf=64):
		super(Disc,self).__init__();
		self.begin = nn.Sequential(nn.Conv2d(in_c,nf,(3,3),stride=1,padding=1), nn.LocalResponseNorm(2), nn.LeakyReLU(negative_slope=0.2));

		self.d_1 = nn.Sequential(\
			nn.Conv2d(nf,nf,(5,5),stride=2,padding=2),\
			nn.LocalResponseNorm(2),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.p_1 = nn.Sequential(\
			nn.Conv2d(nf,2*nf,(3,3),stride=1,padding=1),\
			nn.LocalResponseNorm(2),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.d_2 = nn.Sequential(\
			nn.Conv2d(2*nf,2*nf,(5,5),stride=2,padding=2),\
			nn.LocalResponseNorm(2),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.p_2 = nn.Sequential(\
			nn.Conv2d(2*nf,4*nf,(3,3),stride=1,padding=1),\
			nn.LocalResponseNorm(2),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.d_3 = nn.Sequential(\
			nn.Conv2d(4*nf,4*nf,(5,5),stride=2,padding=2),\
			nn.LocalResponseNorm(2),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.p_3 = nn.Sequential(\
			nn.Conv2d(4*nf,8*nf,(3,3),stride=1,padding=1),\
			nn.LocalResponseNorm(2),\
			nn.LeakyReLU(negative_slope=0.2)\
			);
		self.d_4 = nn.Sequential(\
			nn.Conv2d(8*nf,8*nf,(5,5),stride=2,padding=2),\
			nn.LocalResponseNorm(2),\
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
