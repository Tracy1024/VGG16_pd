##HEADER
#Script for the Autoencoder-Class to call into layer-maker of CNN
#
#Autor: Reza Shah Mohammadi (MRT KIT)

##IMPORTS
import torch
import torchvision
import torch.nn as nn

from LossFunctions import total_variation_loss as tvl

# Class of the autencoder 
class AE(nn.Module):
    #__init__-Method
    def __init__(self, in_channels, **kwargs):
        super(AE, self).__init__()
        self.in_channels = in_channels

        #save channel sizes from previous layer 
        ch_size_1 = in_channels
        ch_size_3 = int(6)
        ch_size_2 = int(ch_size_1 + ch_size_3 / 2)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=ch_size_1, out_channels=ch_size_2,
                        kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=ch_size_2,out_channels=ch_size_3,
                        kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=1) #for compression of H,W set stride=2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ch_size_3, out_channels=ch_size_2, 
                                kernel_size=(3,3), stride=(1,1), padding=(1,1)), #if stride in Maxpool above is 2 then stride here also 2 AND output_padding 1!
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=ch_size_2, out_channels=ch_size_1,
                                kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(True),
            #nn.ConvTranspose2d(in_channels=ch_size_3, out_channels=ch_size_1,
            #                    kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            #nn.Tanh()
        )


    def forward(self, x):
        input_activations = x
        #print(f'Shape before encoder {x.shape}')
        x = self.encoder(x)
        y = x
        tv_loss = tvl(y)
        #print(f'Shape after encoder/before decoder {x.shape}')
        x = self.decoder(x)
        #print(f'Shape after decoder {x.shape}')
        output_activations = x

        return x, y, tv_loss


        