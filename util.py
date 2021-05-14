import numpy as np
import scipy as sp
import cv2
import warnings
import torch
from torchvision.utils import save_image as saveIm
from torch.autograd import grad
from fastai.layers import Flatten
import torch.nn as nn
import random

def getGrad(input_mat):
	if not (input_mat.device=="cpu"):
		input_mat = input_mat.cpu().numpy();
	else:
		input_mat = input_mat.numpy();
	input_mat = input_mat.squeeze().transpose(1,2,0);
	input_mat = input_mat*0.5+0.5;
	if not input_mat.max==0:
		input_mat = 255*input_mat;
	else:
		warnings.warn("Divide by zero occured! Panic!!");
	#RGB to BGR
	input_mat = input_mat[:,:,::-1];
	input_mat = np.array(input_mat, dtype=np.uint8);
	input_mat = cv2.cvtColor(input_mat, cv2.COLOR_BGR2GRAY);
	input_mat = cv2.Laplacian(input_mat, cv2.CV_64F);
	return input_mat;


def genGrad(inp, b, dir, epoch):
	for i in range(b):
		processed = [];
		for inp_ in inp:
			temp = getGrad(inp_[i,:,:,:]);
			processed.append(temp);
		im = cv2.hconcat(processed);
		cv2.imwrite(dir+"epoch"+str(epoch)+"_"+str(i+1)+".png",im);


def getIm(input_mat):
	if not (input_mat.device=="cpu"):
		input_mat = input_mat.cpu().numpy();
	else:
		input_mat = input_mat.numpy();
	input_mat = input_mat.squeeze().transpose(1,2,0);
	input_mat = input_mat*0.5+0.5;
	if not input_mat.max==0:
		input_mat = 255*input_mat;
	else:
		warnings.warn("Divide by zero occured! Panic!!");
	#RGB to BGR
	input_mat = input_mat[:,:,::-1];
	input_mat = np.array(input_mat, dtype=np.uint8);
	return input_mat;

#generate test images
def genIm(inp, b, dir, epoch):
	for i in range(b):
		processed = [];
		for inp_ in inp:
			temp = getIm(inp_[i,:,:,:]);
			processed.append(temp);
		im = cv2.hconcat(processed);
		cv2.imwrite(dir+"epoch"+str(epoch)+"_"+str(i+1)+".png",im);

def smoothenLabel(label, dev):
	noise_ = torch.rand_like(label);
	label = label+(noise_-0.5)*2*dev;
	return label;

def noisy(input_tensor, param):
	noise_ = torch.randn_like(input_tensor);
	input_tensor = input_tensor+noise_*param['std']+param['mean'];
	return input_tensor;

def genParse(inp, b, dir_, epoch):
	 for i in range(b):
	 	inp_ten = inp[i,:,:,:];
	 	inp_ten = torch.unsqueeze(inp_ten, 1);
	 	saveIm(inp_ten, dir_+"epoch"+str(epoch)+"_"+str(i+1)+".png", nrow=5, normalize=True);

def grad_penalty(inp, fake, critic):
	f = Flatten();
	epsilon = np.random.uniform(0,1);
	interpolated = epsilon*inp + (1-epsilon)*fake;
	interpolated.requires_grad = True;
	crit_interp = critic(interpolated);
	gradients = grad(outputs=crit_interp, inputs=interpolated,\
                               grad_outputs=torch.ones_like(crit_interp),\
	                               create_graph=True, retain_graph=True, only_inputs=True)[0];
	gradients = f(gradients);
	gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12);
	result = ((gradients_norm - 1) ** 2);
	return result;

# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Conv2d') != -1:
        # get the number of the inputs
        nn.init.kaiming_uniform_(m.weight);
        m.bias.data.fill_(0);

def extract_patches(input_tensor, target_tensor, size):
	#extracts square patches
	n_row, n_col = input_tensor.shape[2:];
	row_ind = random.randint(0,n_row-size);
	col_ind = random.randint(0,n_col-size);
	return input_tensor[:,:,row_ind:row_ind+size,col_ind:col_ind+size], target_tensor[:,:,row_ind:row_ind+size,col_ind:col_ind+size];