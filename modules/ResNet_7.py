import torch
import torch.nn as nn
from fastai.layers import Flatten

class ResBlock(nn.Module):
	def __init__(self, option, num_chan, num_layers):
		super(ResBlock, self).__init__();
		self.option = option;
		if self.option=='upsample':
			self.scale = nn.UpsamplingNearest2d(scale_factor=2);
			self.begin = nn.Sequential(\
				nn.UpsamplingNearest2d(scale_factor=2),\
				nn.Conv2d(num_chan, num_chan, (3,3), stride=1, padding=1),\
				nn.BatchNorm2d(num_chan),\
				nn.PReLU()\
				);
		elif self.option=='downsample':
			self.scale = nn.AvgPool2d(kernel_size=2, stride=2);
			self.begin = nn.Sequential(\
				nn.Conv2d(num_chan, num_chan, (3,3), stride=2, padding=1),\
				nn.BatchNorm2d(num_chan),\
				nn.PReLU()\
				);
		else:
			self.begin = nn.Sequential(\
				nn.Conv2d(num_chan, num_chan, (3,3), stride=1, padding=1),\
				nn.BatchNorm2d(num_chan),\
				nn.PReLU()\
				);
		self.feat_ext = nn.Sequential(*[nn.Sequential(\
			nn.Conv2d(num_chan, num_chan, (3,3), stride=1, padding=1),\
			nn.BatchNorm2d(num_chan),\
			nn.PReLU()\
			) for i in range(num_layers)]);

	def forward(self, input_tensor):
		temp = input_tensor;
		input_tensor = self.begin(input_tensor);
		input_tensor = self.feat_ext(input_tensor);
		if self.option=='upsample' or self.option=='downsample':
			temp = self.scale(temp);
		input_tensor = input_tensor + temp;
		return input_tensor;

class Encoder(nn.Module):
	def __init__(self, inp_chan, num_chan, num_layers, num_blocks=4, embed_dim=80):
		super(Encoder, self).__init__();
		self.begin = nn.Sequential(\
			nn.Conv2d(inp_chan, num_chan, (3,3), stride=1, padding=1),\
			nn.PReLU()\
			);
		self.main_block = nn.ModuleDict([['layer_%02d'%(i+1), nn.Sequential(\
			ResBlock('downsample', num_chan*(i+1), num_layers),\
			nn.Conv2d(num_chan*(i+1), num_chan*(i+2), (3,3), stride=1, padding=1),\
			nn.BatchNorm2d(num_chan*(i+2)),\
			nn.PReLU()\
			)] for i in range(num_blocks)]);
		self.end_layers = nn.Sequential(\
			nn.Conv2d(num_chan*(num_blocks+1), embed_dim, (1,1))
			);

	def forward(self, input_tensor):
		input_tensor = self.begin(input_tensor);
		for key in self.main_block.keys():
			input_tensor = self.main_block[key](input_tensor);
		input_tensor = self.end_layers(input_tensor);
		input_tensor = input_tensor.view(input_tensor.shape[0],-1);
		return input_tensor;

class Decoder(nn.Module):
	def __init__(self, out_chan, num_chan, num_layers, num_blocks=4, embed_dim=80):
		super(Decoder, self).__init__();
		self.begin = nn.Sequential(\
			nn.Conv2d(embed_dim, num_chan*(num_blocks+1),(1,1)),\
			nn.BatchNorm2d(num_chan*(num_blocks+1)),\
			nn.PReLU()\
			);
		self.main_block = nn.ModuleDict([['layer_%02d'%(i+1), nn.Sequential(\
			ResBlock('upsample', num_chan*(num_blocks+1-i), num_layers),\
			nn.Conv2d(num_chan*(num_blocks+1-i), num_chan*(num_blocks-i), (3,3), stride=1, padding=1),\
			nn.BatchNorm2d(num_chan*(num_blocks-i)),\
			nn.PReLU()\
			)] for i in range(num_blocks)]);
		self.end_layers = nn.Sequential(\
			nn.Conv2d(num_chan, out_chan, (3,3), stride=1, padding=1)\
			);

	def forward(self, input_tensor):
		input_tensor = input_tensor.view(input_tensor.shape[0],-1,1,1);
		input_tensor = self.begin(input_tensor);
		for key in self.main_block.keys():
			input_tensor = self.main_block[key](input_tensor);
		input_tensor = self.end_layers(input_tensor);
		return input_tensor;

class Autoencoder(nn.Module):
	def __init__(self, inp_chan, num_chan, num_layers, num_blocks=4, embed_dim=80):
		super(Autoencoder, self).__init__();
		self.encoder = Encoder(inp_chan, num_chan, num_layers, num_blocks, embed_dim);
		self.decoder = Decoder(inp_chan, num_chan, num_layers, num_blocks, embed_dim);

	def forward(self, input_tensor):
		outputs = {};
		enc_out = self.encoder(input_tensor);
		outputs['feat'] = enc_out;
		dec_out = self.decoder(enc_out);
		outputs['dec_out'] = dec_out;
		return outputs;

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
