import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_ as clipGrad
from torchvision.utils import save_image as saveIm
from torch.optim.lr_scheduler import StepLR
from torch_two_sample.statistics_diff import MMDStatistic as MMD
from torch.distributions.normal import Normal 
from torch.distributions.laplace import Laplace
import os
from tqdm import tqdm
from PIL import Image
from torchsummary import summary
import numpy as np
import scipy as sp
import cv2
from skimage.util import img_as_ubyte
import warnings
from itertools import chain,cycle
from math import log10, sqrt
from util import *
from shutil import rmtree


#For the first time only
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.manual_seed(0);
torch.backends.cudnn.deterministic = True;
torch.backends.cudnn.benchmark = True;
np.random.seed(0);
torch.autograd.set_detect_anomaly(True);

if os.path.exists("tensorboard_adv"):
	rmtree('tensorboard_adv');

writer = SummaryWriter("tensorboard_adv");

#Creating Necessary Folder
model_save = "Trained_Models/";
if not os.path.exists(model_save):
	os.mkdir(model_save);

scale_steps = 2;
embed_dim = 60;
model_params = {'h2l': [64,16], 'enc_mix': [64], 'dec': [64], 'disc': [3,64], 'cls':[]};
learning_rate = {'enc': 1e-6, 'enc_mix': 1e-4, 'dec': 1e-4, 'disc': 4e-4, 'cls': 4e-6};
data_root = '/datasets/Saurabh/GameSR/Datasets/';
train_data_list = {'inp': '/datasets/Saurabh/GameSR/Datasets/LR_train.csv', 'targ': '/datasets/Saurabh/GameSR/Datasets/HR_train.csv'};
strengths = {'train': 182866, 'test': 15000};
test_data_list = {'inp': '/datasets/Saurabh/GameSR/Datasets/LR_test.csv', 'targ': '/datasets/Saurabh/GameSR/Datasets/LR_test.csv'};
lr_decay = {'enc': 2, 'enc_mix': 2, 'dec': 2, 'disc': 2, 'cls': 2};
batch_size = {'train': 13, 'test': 5};
epoch = 1;
alpha = 0.3;
__checkpoint__ = 'checkpoint_train4x4_gradual.pth';

#Cuda stuffs
cuda_devices = [0];
device = 'cuda:0' if torch.cuda.is_available() else 'cpu';

#Loading the data
from Datasets.CustomDataloader import loadData
transform = {'inp_deg': transforms.Compose([transforms.Resize([16,16]),transforms.ToTensor(), transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])]),\
'inp_ds': transforms.Compose([transforms.Resize([25,25]), transforms.CenterCrop(16), transforms.ToTensor(), transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])]),\
'targ': transforms.Compose([transforms.Resize([100,100]), transforms.CenterCrop(64), transforms.ToTensor(), transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])};
trainset = loadData(data_root, strengths['train'], train_data_list, transform);
trainloader = DataLoader(trainset, batch_size=batch_size['train'], shuffle=True, drop_last=True ,num_workers=8, pin_memory=True);
testset = loadData(data_root, strengths['test'],test_data_list, transform);
testloader = DataLoader(testset, batch_size=batch_size['test'], shuffle=True, drop_last=True, num_workers=8, pin_memory=True);

#Defining a model
from modules.ResNet_4x4 import Enc, Dec, Disc
from module3 import H2L
from modules.classifier import Cls

Models = {\
'h2l': H2L(*model_params['h2l']),\
'enc_mix': Enc(*model_params['enc_mix']),\
'dec': Dec(*model_params['dec']),\
'disc': Disc(*model_params['disc'])\
};

#In hope that one day, however distant that day might be, I'll get to run this models on more than one GPU!! :-(

if torch.cuda.device_count()>=1:
	print("Using {} GPUs!! Yeaaaaaaaa!!!\n".format(len(cuda_devices)));
	if len(cuda_devices)==1:
		print("No, I'm not weak at Grammar, I just didn't feel like writing another line to appease a Grammar-Nazi.\n But wait a sec! I did write this line though, didn't I?\n Well, you can amuse yourself over that. Consider this a consolation.");
	for key in Models.keys():
		Models[key] = nn.DataParallel(Models[key], device_ids=cuda_devices);
for key in Models.keys():	
	Models[key] = Models[key].to(device);
	# Models[key].apply(weights_init_uniform_rule);

#Loading parameters for H2L
checkpt_h2l = torch.load(model_save+'/checkpoint_h2l2new2.pth');
Models['h2l'].load_state_dict(checkpt_h2l['models_state']['gen'], strict=False);
Models['h2l'].eval();
Models['h2l'].requires_grad = False;

#Defining optimizers
Optimizers = {\
'enc_mix+dec': optim.Adam(chain(Models['enc_mix'].parameters(),Models['dec'].parameters()), betas=(0, 0.5), lr=learning_rate['dec']),\
'enc': optim.Adam(Models['enc_mix'].parameters(), betas=(0, 0.5), lr=learning_rate['enc']),\
'disc': optim.Adam(Models['disc'].parameters(), betas=(0, 0.5), lr=learning_rate['disc'])\
};

#Schedulers
Schedulers = {\
'enc_mix+dec': StepLR(Optimizers['enc_mix+dec'], step_size=lr_decay['dec'], gamma=0.2),\
'enc': StepLR(Optimizers['enc'], step_size=lr_decay['enc'], gamma=0.2),\
'disc': StepLR(Optimizers['disc'], step_size=lr_decay['disc'], gamma=0.2)\
};

#Losses
from Losses.featureLoss import featLoss2
from fml.nn import SinkhornLoss

f = featLoss2(5).to(device);
mse_loss = nn.MSELoss().to(device);
l1_loss = nn.L1Loss().to(device);
ot_loss = SinkhornLoss().to(device);
bce_loss = nn.BCEWithLogitsLoss();

#Continuing Training if records found
if os.path.exists(model_save+__checkpoint__):
	checkpoint_ = torch.load(model_save+__checkpoint__);
	print("Continuing the training...");
	#Load model state dict
	for key in Models.keys():
		if not key == 'h2l':
			Models[key].load_state_dict(checkpoint_['models_state'][key]);
	#Loading optimizer state dict
	for key in Optimizers.keys():
		if not key == 'h2l':
			Optimizers[key].load_state_dict(checkpoint_['optimizers_state'][key]);
	#Loading scheduler state dict
	for key in Schedulers.keys():
		if not key == 'h2l':
			Schedulers[key].load_state_dict(checkpoint_['scheduler_state'][key]);
	epoch = checkpoint_['epoch'];

testIter = cycle(testloader);

#The sampler
m_train = Normal(torch.zeros([batch_size['train'],16,1,1], device=device, dtype=torch.float, requires_grad=False), torch.ones([batch_size['train'],16,1,1], device=device, dtype=torch.float, requires_grad=False));
m_test = Normal(torch.zeros([batch_size['test'],16,1,1], device=device, dtype=torch.float, requires_grad=False), torch.ones([batch_size['test'],16,1,1], device=device, dtype=torch.float, requires_grad=False));

while True:
	print("Epoch {} training.\n".format(epoch));
	for key in Models.keys():
		if not key == 'h2l':
			Models[key] = Models[key].train();

	counter = 1;

	for data in tqdm(trainloader):
		inp_ds =data['inp_ds']; targ = data['targ']; targ_im = data['inp_deg'];
		inp_ds = inp_ds.to(device); 
		z = m_train.sample().to(device)
		inp_deg = Models['h2l'](inp_ds,z).clamp(-1,1).to(device).detach();
		inp_mix = alpha*inp_ds + (1-alpha)*inp_deg;
		targ = targ.to(device);
		targ_im = targ_im.to(device);

		feat_mix = Models['enc_mix'](inp_mix);
		feat_ds = Models['enc_mix'](inp_ds);
		feat_deg = Models['enc_mix'](inp_deg);		

		fake_ds = Models['dec'](feat_ds);
		fake_deg = Models['dec'](feat_deg);
		fake_mix = Models['dec'](feat_mix);
		fake_targ = Models['dec'](Models['enc_mix'](targ_im)).detach();

		
		#Training critic_sharp
		Optimizers['disc'].zero_grad();
		real_out = Models['disc'](targ);
		fake_out = Models['disc'](fake_mix.detach());
		critic_loss = fake_out.mean() - real_out.mean() + 10*grad_penalty(targ,fake_mix.detach(),Models['disc']).mean();
		writer.add_scalar('Train/Critic_Loss', critic_loss.item(), counter);
		critic_loss.backward();
		Optimizers['disc'].step();
		del real_out, fake_out, critic_loss;

		if counter%5==0: 
			#Train enc-dec
			Optimizers['enc_mix+dec'].zero_grad();


			gan_out = Models['disc'](fake_mix).mean();
			gan_loss = -gan_out;
			writer.add_scalar('Train/GAN_Loss', gan_loss.item(), counter);

			l1_hr = l1_loss(fake_mix,targ);
			writer.add_scalar('Train/L1_Loss', l1_hr.item(), counter);

			f_hr = f(fake_mix, targ);
			writer.add_scalar('Train/feat_Loss', f_hr.item(), counter);


			sr_loss = f_hr + 5e-1*l1_hr + 5e-2*gan_loss;
			
			sr_loss.backward();

			Optimizers['enc_mix+dec'].step();

			del gan_out, gan_loss, l1_hr, f_hr, sr_loss;

			if epoch>=2: 
				Optimizers['enc'].zero_grad();

				feat_ds = Models['enc_mix'](inp_ds);
				feat_deg = Models['enc_mix'](inp_deg);	

				b,c,h,w = feat_deg.shape;	

				feat_ds_loss = ot_loss(feat_ds.view(b,h*w,c),feat_mix.view(b,h*w,c).detach()).mean();
				feat_deg_loss = ot_loss(feat_deg.view(b,h*w,c),feat_mix.view(b,h*w,c).detach()).mean();

				feat_loss = alpha*feat_ds_loss + (1-alpha)*feat_deg_loss;
				writer.add_scalar('Train/OT_Loss', feat_loss.item(), counter);
				feat_loss.backward();

				Optimizers['enc'].step();

				del feat_ds_loss, feat_deg_loss, feat_loss;
		
		
		counter += 1;

		#Summary for Tensorboard
		writer.add_images('Train/Input_DS', 0.5*inp_ds+0.5, counter);
		writer.add_images('Train/Input_Deg', 0.5*inp_deg+0.5, counter);
		writer.add_images('Train/Input_Mix', 0.5*inp_mix+0.5, counter);
		writer.add_images('Train/Target_Deg', 0.5*targ_im+0.5, counter);
		writer.add_images('Train/Target_SR', 0.5*fake_targ.clamp(-1,1)+0.5, counter);
		writer.add_images('Train/Output_DS', 0.5*fake_ds.clamp(-1,1)+0.5, counter);
		writer.add_images('Train/Output_Deg', 0.5*fake_deg.clamp(-1,1)+0.5, counter);
		writer.add_images('Train/Output_Mix', 0.5*fake_mix.clamp(-1,1)+0.5, counter);

		del inp_ds, inp_deg, inp_mix, targ_im, targ, feat_ds, feat_deg, feat_mix, fake_ds, fake_deg, fake_mix, fake_targ;

	for key in Schedulers.keys():
		Schedulers[key].step();
	
	epoch += 1;
	
	checkpoint = {\
	'models_state':{key:Models[key].state_dict() for key in Models.keys() if not key =='h2l'},\
	'optimizers_state':{key:Optimizers[key].state_dict() for key in Optimizers.keys()},\
	'scheduler_state':{key:Schedulers[key].state_dict() for key in Schedulers.keys()},\
	'epoch':epoch\
	};

	torch.save(checkpoint, model_save+__checkpoint__);

	#Testing the Model
	for key in Models.keys():
		Models[key] = Models[key].eval();
	with torch.no_grad():
		test_data = next(testIter);	
	
		test_ip = test_data['inp_deg'];
		test_op = Models['dec'](Models['enc_mix'](test_ip));

		#Summary
		writer.add_images('Test/Input', 0.5*test_ip+0.5, epoch-1);
		writer.add_images('Test/Output', 0.5*test_op.clamp(-1,1)+0.5, epoch-1);
		
		del test_data, test_ip, test_op; 


