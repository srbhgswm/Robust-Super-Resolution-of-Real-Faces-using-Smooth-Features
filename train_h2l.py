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

torch.manual_seed(0);
torch.backends.cudnn.deterministic = True;
torch.backends.cudnn.benchmark = False;
np.random.seed(0);
torch.autograd.set_detect_anomaly(True);

if os.path.exists("tensorboard_h2l_new"):
	rmtree('tensorboard_h2l_new');

writer = SummaryWriter("tensorboard_h2l_new");

#Creating Necessary Folder
model_save = "Trained_Models/";
if not os.path.exists(model_save):
	os.mkdir(model_save);

learning_rate = {'gen': 1e-4, 'disc': 4e-4};
data_root = 'Datasets/';
train_data_list = {'inp': 'Datasets/LR_train.csv', 'targ': 'Datasets/HR_train.csv'};
strengths = {'train': 182866, 'test': 15000};
test_data_list = {'inp': 'Datasets/LR_test.csv', 'targ': 'Datasets/LR_test.csv'};
lr_decay = {'gen': 1, 'disc': 1};
batch_size = {'train': 20, 'test': 5};
epoch = 1;
__checkpoint__ = 'checkpoint_h2l2_bulat.pth';

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
from modules.h2l_modules import Gen, Disc


Models = {\
'gen': Gen(64),\
'disc': Disc(64)\
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

#Defining optimizers
Optimizers = {\
'gen': optim.Adam(Models['gen'].parameters(), betas=(0.5, 0.999), lr=learning_rate['gen']),\
'disc': optim.SGD(Models['disc'].parameters(), momentum=0.5, lr=learning_rate['disc'])\
};

#Schedulers
Schedulers = {\
'gen':StepLR(Optimizers['gen'], step_size=lr_decay['gen'], gamma=0.5),\
'disc':StepLR(Optimizers['disc'], step_size=lr_decay['disc'], gamma=0.5)\
};

#Losses
from losses import featLoss
from ContextualLoss import Contextual_Loss as CL
f = featLoss(6).to(device);
nll_loss = nn.NLLLoss().to(device);
l1_loss = nn.SmoothL1Loss().to(device);
mse_loss = nn.MSELoss().to(device);
bce_stable = nn.BCEWithLogitsLoss().to(device);

#Continuing Training if records found
if os.path.exists(model_save+__checkpoint__):
	checkpoint_ = torch.load(model_save+__checkpoint__);
	print("Continuing the training...");
	#Load model state dict
	for key in Models.keys():
		Models[key].load_state_dict(checkpoint_['models_state'][key]);
	#Loading optimizer state dict
	for key in Optimizers.keys():
		Optimizers[key].load_state_dict(checkpoint_['optimizers_state'][key]);
	#Loading scheduler state dict
	for key in Schedulers.keys():
		Schedulers[key].load_state_dict(checkpoint_['scheduler_state'][key]);
	epoch = checkpoint_['epoch'];

testIter = cycle(testloader);

m_train = Normal(torch.zeros([batch_size['train'],16], device=device, dtype=torch.float, requires_grad=False), torch.ones([batch_size['train'],16], device=device, dtype=torch.float, requires_grad=False));
m_test = Normal(torch.zeros([batch_size['test'],16], device=device, dtype=torch.float, requires_grad=False), torch.ones([batch_size['test'],16], device=device, dtype=torch.float, requires_grad=False));

while True:
	print("Epoch {} training.\n".format(epoch));
	for key in Models.keys():
		Models[key] = Models[key].train();

	counter = 1;

	for data in tqdm(trainloader):
		inp_deg = data['inp_deg']; inp_ds = data['inp_ds']; targ = data['targ'];
		dummy = m_train.sample();

		inp_deg = inp_deg.to(device);
		inp_ds = inp_ds.to(device);
		targ = targ.to(device);
		dummy = dummy.to(device);
		
		fake = Models['gen'](targ,dummy);

		#Training the disc
		Optimizers['disc'].zero_grad();
		fake_out = Models['disc'](fake.detach());
		real_out = Models['disc'](inp_deg);
		critic_loss = fake_out.mean() - real_out.mean() + 10*grad_penalty(fake.detach(),inp_deg,Models['disc']).mean();
		writer.add_scalar('Train/Disc', critic_loss.item(), counter);
		critic_loss.backward();
		Optimizers['disc'].step();
		del fake_out, real_out, critic_loss;		
		
		#Train Gen
		if counter%5 == 0:
			Optimizers['gen'].zero_grad();
			fake_out = Models['disc'](fake);
			pixel_loss = mse_loss(fake, inp_ds);
			writer.add_scalar('Train/Pixel_Loss', pixel_loss.item(), counter);
			gan_loss = -fake_out.mean();
			writer.add_scalar('Train/GAN_Loss', gan_loss.item(), counter);

			loss = 0.8*pixel_loss + gan_loss;
			loss.backward();
			Optimizers['gen'].step();

			del fake_out,pixel_loss,gan_loss,loss;

		counter += 1;

		#Summary for Tensorboard
		if counter%10 == 0:
			writer.add_images('Train/Input_DS', 0.5*inp_ds+0.5, counter);
			writer.add_images('Train/Input_Deg', 0.5*inp_deg+0.5, counter);
			writer.add_images('Train/Output', 0.5*fake.clamp(-1,1)+0.5, counter);

		del inp_ds, inp_deg, fake, targ, dummy;

	for key in Schedulers.keys():
		Schedulers[key].step();
	
	epoch += 1;
	
	checkpoint = {\
	'models_state':{key:Models[key].state_dict() for key in Models.keys()},\
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

		test_deg = test_data['inp_deg'].to(device);
		test_ds = test_data['inp_ds'].to(device);
		test_targ = test_data['targ'].to(device);
		dummy = m_test.sample().to(device);

		test_op = Models['gen'](test_targ,dummy);
		
		#Summary
		writer.add_images('Test/Input_DS', 0.5*test_ds+0.5, epoch-1);
		writer.add_images('Test/Input_Deg', 0.5*test_deg+0.5, epoch-1);
		writer.add_images('Test/Output', 0.5*test_op.clamp(-1,1)+0.5, epoch-1);


