import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      
import matplotlib.pyplot as plt
from load_data import load_data
import numpy as np


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()        
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid(),      
        )

    def forward(self, x):
        x=x.view(-1,28*28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)    
        decoded=decoded.view(-1,1,28,28)
        return decoded

