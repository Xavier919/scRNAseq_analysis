import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_size):  
        super(Autoencoder, self).__init__()
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 32),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, self.input_size),
            nn.Sigmoid()  # or another suitable activation function
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x