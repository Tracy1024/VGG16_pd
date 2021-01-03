##HEADER
#Script for the Losses
#
#Autor: Reza Shah Mohammadi (MRT KIT)

##IMPORTS
import torch
import torchvision
import torch.nn as nn


def total_variation_loss(img):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)
