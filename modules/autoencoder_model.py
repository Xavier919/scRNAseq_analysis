import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, dropout_rate=0.2):  
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(13492, 4096),
            nn.GELU(True),
            #nn.Dropout(dropout_rate), 
            nn.Linear(4096, 1024),
            nn.GELU(True),
            #nn.Dropout(dropout_rate),  
            nn.Linear(1024, 256),
            nn.GELU(True),
            #nn.Dropout(dropout_rate),
            nn.Linear(256, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.GELU(True),
            #nn.Dropout(dropout_rate),  
            nn.Linear(256, 1024),
            nn.ReLU(True),
            #nn.Dropout(dropout_rate),  
            nn.Linear(1024, 4096),
            nn.GELU(True),
            #nn.Dropout(dropout_rate),  
            nn.Linear(4096, 13492),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x