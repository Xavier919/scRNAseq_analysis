import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):  
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(13492, 4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
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
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, 13492),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x