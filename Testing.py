##HEADER
#CNN (VGG16) for detection of pedestrians
#Start of Coding: 03.11.2020
#...
#Autor: Reza Shah Mohammadi (MRT KIT)
#

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils import tensorboard
from skimage import io
from PIL import Image
from pathlib import Path




#ACCURACY CHECK
def check_accuracy(loader, model, device, train_ae=False, save_feature_maps=False):
    num_correct = 0 
    num_samples = 0
    model.eval() #let the model know that this is evaluation mode(?)


    with torch.no_grad(): #so no calculations of gradients are done
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            #x = x.reshape(x.shape[0], -1)

            if not train_ae:
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
            else:
                before_ae, in_ae, tv_loss, after_ae, scores, scores_ae = model(x)
                _, predictions = scores_ae.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

                
                if save_feature_maps:
                    feature_map_dir = "./feature_maps"
                    Path(feature_map_dir).mkdir(parents=True, exist_ok=True)
                    
                    features_bs, features_c, h_features, w_features = in_ae.size()
                    
                    #feature_bs = 1

                    for batch_idx in np.arange(features_bs):
                        fig, axarr = plt.subplots()
                        for feature_idx in np.arange(features_c):
                            vals = in_ae[batch_idx,feature_idx,:,:].cpu()
                            x = np.arange(-0.5,w_features,1)
                            y = np.arange(-0.5,h_features,1)
                            axarr.pcolormesh(x,y,vals)
                            
                            pngname = feature_map_dir + '/' + f'batch{feature_idx}_feature{feature_idx}'
                            plt.savefig(pngname, dpi=150, format='png', pad_inches=0.1)
                            
                            #pdfname = feature_map_dir + subdir + '/' + f'batch{feature_idx}_feature{feature_idx}'
                            #plt.savefig(pdfname, dpi=150, format='pdf', pad_inches=0.1)
                        

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()


