# Robust-Super-Resolution-of-Real-Faces-using-Smooth-Features
The official implementation of our work published in Adversarial Robustness in the Real World workshop, ECCV 2020.
Check out the [paper](https://arxiv.org/abs/2011.02427#:~:text=Real%20low%2Dresolution%20(LR),kernels%20and%20signal%2Dindependent%20noises.) for more details.

# To train the model
*Compile the files 'LR_train.csv', 'LR_test.csv', 'HR_train.csv', 'HR_test.csv' in the Datasets folder with complete location of all the LR and HR images you'll be using for training and testing.
* Make sure you have all the modules installed.
* Run 'train_h2l.py' to train the degradation network first.
* Run 'train.py' afterwards.

# For testing
Put your LR images in the 'LR/' folder and run the 'test.py' code.

This repository contains only the bare minimum files required for training. Feel free to raise an issue if you feel I've missed a required file.

#BibTex
If you use this code in your research, please cite the following paper

@InProceedings{10.1007/978-3-030-66415-2_11,
author="Goswami, Saurabh
and Aakanksha
and Rajagopalan, A. N.",
editor="Bartoli, Adrien
and Fusiello, Andrea",
title="Robust Super-Resolution of Real Faces Using Smooth Features",
booktitle="Computer Vision -- ECCV 2020 Workshops",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="169--185",
abstract="Real low-resolution (LR) face images contain degradations which are too varied and complex to be captured by known downsampling kernels and signal-independent noises. So, in order to successfully super-resolve real faces, a method needs to be robust to a wide range of noise, blur, compression artifacts etc. Some of the recent works attempt to model these degradations from a dataset of real images using a Generative Adversarial Network (GAN). They generate synthetically degraded LR images and use them with corresponding real high-resolution (HR) image to train a super-resolution (SR) network using a combination of a pixel-wise loss and an adversarial loss. In this paper, we propose a two module super-resolution network where the feature extractor module extracts robust features from the LR image, and the SR module generates an HR estimate using only these robust features. We train a degradation GAN to convert bicubically downsampled clean images to real degraded images, and interpolate between the obtained degraded LR image and its clean LR counterpart. This interpolated LR image is then used along with it's corresponding HR counterpart to train the super-resolution network from end to end. Entropy Regularized Wasserstein Divergence is used to force the encoded features learnt from the clean and degraded images to closely resemble those extracted from the interpolated image to ensure robustness.",
isbn="978-3-030-66415-2"
}

