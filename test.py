import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from torchvision.utils import save_image,make_grid

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height));
    dst.paste(im1, (0, 0));
    dst.paste(im2, (im1.width, 0));
    return dst;

#Preparing the model
device = "cuda:0" if torch.cuda.is_available() else "cpu";
from modules.ResNet_4x4 import Enc, Dec
enc = Enc(64);
gen = Dec(64);
cuda_devices = [0];
if torch.cuda.device_count()>=1:
	enc = nn.DataParallel(enc, device_ids=cuda_devices);
	gen = nn.DataParallel(gen, device_ids=cuda_devices);
enc = enc.to(device);
gen = gen.to(device);

enc.eval();
gen.eval();

checkpoint = torch.load("Trained_Models/checkpoint_train4x4_gradual.pth");
enc.load_state_dict(checkpoint["models_state"]['enc_mix']);
gen.load_state_dict(checkpoint["models_state"]['dec']);

print("\nModel Loaded.\n");

#Images for test
im_list = 'LR/';


from PIL import Image
from torchvision import transforms

t = transforms.Compose([transforms.Resize([16,16]), transforms.ToTensor(), transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])]);
t_out = transforms.Compose([transforms.ToPILImage()]);

from tqdm import tqdm

#Directories
if not os.path.exists('Results/'):
	os.makedirs('Results/');

#Clean images
print('Processing...');
with torch.no_grad():
	for i in tqdm(os.listdir(im_list)):
		im = Image.open(im_list+i).convert('RGB');

		im = t(im);
		im = im.to(device);
		im = im.unsqueeze(0);
		out = gen(enc(im)).clamp(-1,1).detach().cpu();
		out = t_out(0.5*out[0,:,:,:]+0.5);
		out.save('Results/'+i);
	

print("\nDone!\n");