##HEADER
#Script to call for Dataset imports into CNN
#
#Autor: Reza Shah Mohammadi (MRT KIT)

##IMPORTS
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import cv2

#root=./data/INRIAPerson/

class INRIAPersonDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.neg_list = pd.read_csv(root_dir + mode + '/neg.lst', index_col=False, header=None)
        self.pos_list = pd.read_csv(root_dir + mode + '/pos.lst', index_col=False, header=None)
        self.file_list = pd.concat([self.neg_list,self.pos_list], ignore_index=True)
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        img_path = os.path.join(self.root_dir, self.file_list.iloc[index,0])
        image = io.imread(img_path)
        #image = cv2.imread(img_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if '/neg/' in img_path:
            y_label = torch.tensor(int(0))
        elif '/pos/' in img_path:
            y_label = torch.tensor(int(1))

        if self.transform:
            image = self.transform(image)
            #augmented = self.transform(image=image)
            #image = augmented['image']

        return(image, y_label)
