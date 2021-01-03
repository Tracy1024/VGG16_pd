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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils import tensorboard
from skimage import io
from PIL import Image
from PedestrianDatasets import INRIAPersonDataset

## VGG16 NETWORK IMPLEMENTATION
# Architechture of network --> Autoencoders need to be added
VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


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

    # forward-Method
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

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


# Set device
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
learning_rate = 0.0001
batch_size = 32
num_epochs = 5
classes = ['No Pedestrian', 'Pedestrian']

##LOAD AND PREPARE DATA
# Transformations
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256)),
    transforms.CenterCrop(256),
    transforms.RandomCrop(img_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

normalizes = transforms.Normalize(norm_mean, norm_std)
valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.TenCrop(img_size, vertical_flip=False),
    transforms.Lambda(lambda crops: torch.stack([normalizes(transforms.ToTensor()(crop)) for crop in crops])),
])

# img_transforms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((img_size,img_size)),
#     #transforms.CenterCrop((img_size,img_size)),
#     transforms.ToTensor()
#     ])

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
model = VGG_net(in_channels=in_channels, num_classes=num_classes).to(
    device)  # in_channels=input_size, num_classes=num_classes

# LOSS AND OPTIMIZER
# Loss and optimizer --> check: Is the CrossEntropyLoss applicable with the 0/1 classifier?
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# LOGGING
writer = tensorboard.SummaryWriter(f'runs/INRIA/tryingout_tensorboard')

for epoch in range(num_epochs):
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
        writer.add_histogram('fcs', model.fcs[6].weight)

        # Use SummaryWriter for loss and accuracy
        writer.add_scalar('Training loss', loss, global_step=step)
        writer.add_scalar('Training Accuracy', running_train_acc, global_step=step)

        # writer.add_embedding(features, metadata=class_labels, label_img = data, global_step=step)
        step += 1

    writer.add_hparams({'lr': learning_rate, 'bsize': batch_size},
                       {'accuracy': sum(accuracies) / len(accuracies), 'loss': sum(losses) / len(losses)})


# Check accuracy on training & test to see how good the model is

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # let the model know that this is evaluation mode(?)

    with torch.no_grad():  # so no calculations of gradients are done
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            if loader == test_loader:
                bs, ncrpos, c, h, w = x.size()
                scores = model(x.view(-1, c, h, w))
                scores = scores.view(bs, ncrpos, -1).mean(1)
            else:
                scores = model(x)

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


print("Cheking accuracy on training data")
check_accuracy(train_loader, model)
print("Checking accuracy on test data")
check_accuracy(test_loader, model)
