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

#import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils import tensorboard
from skimage import io
from PIL import Image

# NETWORK TRAINING
def train_network(model, criterion, optimizer, train_loader, num_epochs, device):

    for epoch in range(num_epochs):
        losses = []
        accuracies = []
        print("Current epoch: " , (epoch+1) , " of " , num_epochs)

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            #print(data.shape)
            targets = targets.to(device=device)

            # Get to correct shape
            #data = data.reshape(data.shape[0], -1)
            #print(data.shape)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            losses.append(loss.item())

            # backward
            optimizer.zero_grad() #set gradients to zero for each batch so it doesn't store the back prop calcs from previous for props
            loss.backward()
            
            # gradient descent or adam step
            optimizer.step()

            # Calculate 'running' training accuracy
            _, predictions = scores.max(1)
            num_correct = (predictions == targets).sum()
            running_train_acc = float(num_correct)/float(data.shape[0])
            accuracies.append(running_train_acc)


#ACCURACY CHECK
def check_accuracy(loader, model, device):
    num_correct = 0 
    num_samples = 0
    model.eval() #let the model know that this is evaluation mode(?)

    with torch.no_grad(): #so no calculations of gradients are done
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            #x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()


