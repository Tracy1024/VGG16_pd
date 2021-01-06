##HEADER
# CNN (VGG16) for detection of pedestrians
# Start of Coding: 03.11.2020
# 1st goal: Apply transfer learning, fine tuning and check accuracy,...
# 2nd goal: Train network on MRT-Knecht server, check accuracy
# 3rd goal: Add autoencoders and view results
# ...
# Autor: Reza Shah Mohammadi (MRT KIT)
#

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import albumentations as A

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import cv2

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils import tensorboard
from skimage import io
from PIL import Image
from PedestrianDatasets import INRIAPersonDataset
from albumentations.pytorch import ToTensorV2

## VGG16 NETWORK IMPLEMENTATION
# Architechture of network --> Autoencoders need to be added
VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

# parameters from pretraining
pre_param = './vgg16_bn.pth'

# Then flatten and 4096x4096x1000 linear layers

# Class of the network
class VGG_net(nn.Module):
    # __init__-Method
    def __init__(self, in_channels, num_classes):  #
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.make_conv_layers(VGG16)

        # Classifier layers (more compact when implemented in an nn.Sequential)
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )


	#if not pretrained:
       	    #self._initialize_weights()

    # forward-Method
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x
	
    # parameters intialize
   # def _initialize_weights(self) -> None:
   #     for m in self.modules():
   #         if isinstance(m, nn.Conv2d):
   #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
   #             if m.bias is not None:
   #                 nn.init.constant_(m.bias, 0)
   #         elif isinstance(m, nn.BatchNorm2d):
   #             nn.init.constant_(m.weight, 1)
   #             nn.init.constant_(m.bias, 0)
   #         elif isinstance(m, nn.Linear):
   #             nn.init.normal_(m.weight, 0, 0.01)
   #             nn.init.constant_(m.bias, 0)    		

    # Layermaker --> Here we need to implement the autoencoders
    def make_conv_layers(self, architechture):
        layers = []
        in_channels = self.in_channels

        for x in architechture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

def VGG_model(pretrained: bool): 

    model = VGG_net(in_channels=in_channels, num_classes=num_classes).to(
    device)
    if pretrained:
        pre_state_dict = torch.load(pre_param)
        model_dict =  model.state_dict()
        state_dict = {k:v for k,v in pre_state_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model


# Check accuracy on training & test to see how good the model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # let the model know that this is evaluation mode(?)

    with torch.no_grad():  # so no calculations of gradients are done
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            #if loader == test_loader:
            #    bs, ncrpos, c, h, w = x.size()
            #    scores = model(x.view(-1, c, h, w))
            #    scores = scores.view(bs, ncrpos, -1).mean(1)
            #else:
            #    scores = model(x)
           
            scores = model(x)

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            acc = float(num_correct) / float(num_samples) * 100

        #print(f'Got {num_correct} / {num_samples} with accuracy {acc}')

    model.train()
    return acc


# Set device
torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------------------------------------------------------------
'''
##NETWORK TEST
model = VGG_net(in_channels=3, num_classes=1000)
x = torch.randn(1,3,224, 224).to(device)
print(model(x).shape)
'''
# --------------------------------------------------------------------------------------


# Hyperparameters
img_size = 224  # --> change later
num_classes = 2
in_channels = 3
learning_rate = 0.01
batch_size = 20
num_epochs = 10
classes = ['No Pedestrian', 'Pedestrian']

##LOAD AND PREPARE DATA
# Transformations
# parameters for normalization from ImageNet
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.RandomCrop(img_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
#train_transform =A.Compose([
#    A.Resize(256,256),
#    A.RandomCrop(img_size,img_size),
#    #A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.5),
#    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20,p=0.5),
#    A.HorizontalFlip(p=0.5),
#    A.Normalize(norm_mean, norm_std),
#    ToTensorV2()
#    ])

valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
#valid_transform = A.Compose([
#    A.Resize(img_size,img_size),
#    A.Normalize(norm_mean, norm_std),
#    ToTensorV2()
#    ])

# Training data
train_dataset = INRIAPersonDataset(root_dir='~/Documents/INRIAPerson/', mode='Train', transform=train_transform)
# Test data
test_dataset = INRIAPersonDataset(root_dir='~/Documents/INRIAPerson/', mode='Test', transform=valid_transform)

# for img, label in train_dataset:
#    print(img.shape, label)

##NETWORK TRAINING WITH LOGGING

# -->Here maybe (randomized) hyperparameter search

# helper variable step
step = 0
# DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = VGG_model(True)

# LOSS AND OPTIMIZER
# Loss and optimizer --> check: Is the CrossEntropyLoss applicable with the 0/1 classifier?
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# LOGGING
writer = tensorboard.SummaryWriter(f'runs/INRIA/tryingout_tensorboard')

for epoch in range(num_epochs):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    losses = []
    accuracies = []

    print("Current epoch: ", (epoch + 1), " of ", num_epochs)

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        # print(data.shape)
        targets = targets.to(device=device)

        # Get to correct shape
        # data = data.reshape(data.shape[0], -1)
        # print(data.shape)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()  # set gradients to zero for each batch so it doesn't store the back prop calcs from previous for props
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        # Calculate 'running' training accuracy
        _, predictions = scores.max(1)
        num_correct = (predictions == targets).sum()
        running_train_acc = float(num_correct) / float(data.shape[0])
        accuracies.append(running_train_acc)

        # Make visualization of the classification
        features = data.reshape(data.shape[0], -1)
        class_labels = [classes[label] for label in predictions]

        # Make image grid for visualization
        # img_grid = torchvision.utils.make_grid(data)
        # writer.add_image('INRIA_imgs', img_grid)

        # Histogram to see if weights change in the layers
        #writer.add_histogram('fcs', model.fcs[6].weight)

        # Use SummaryWriter for loss and accuracy
        #writer.add_scalar('Training loss', loss, global_step=step)
        #writer.add_scalar('Training Accuracy', running_train_acc, global_step=step)

        # writer.add_embedding(features, metadata=class_labels, label_img = data, global_step=step)
        #step += 1
 
    acc_test = check_accuracy(test_loader, model)

    print(f'learning_rate: {learning_rate}')
    if acc_test > 95:
        learning_rate = 0.0001
    elif acc_test > 90:
        learning_rate = 0.001
    else: 
        learning_rate = 0.01
    
    writer.add_scalar('Training Accuracy', sum(accuracies) / len(accuracies) * 100, global_step=epoch)
    writer.add_scalar('Test Accuracy', acc_test, global_step=epoch)
    writer.add_scalar('Training Loss', sum(losses) / len(losses), global_step=epoch)
    writer.add_hparams({'lr': learning_rate, 'bsize': batch_size},
                       {'accuracy': sum(accuracies) / len(accuracies), 'loss': sum(losses) / len(losses)})
    
    print(f'Cheking accuracy on training data: {sum(accuracies) / len(accuracies) * 100}')  
    print(f'Cheking accuracy on test data: {acc_test}')

