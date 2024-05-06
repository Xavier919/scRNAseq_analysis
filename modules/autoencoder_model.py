import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, dropout_rate=0.2):  # dropout_rate is an adjustable parameter
        super(Autoencoder, self).__init__()

        # Encoder with Dropout after each ReLU
        self.encoder = nn.Sequential(
            nn.Linear(13492, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),  # Adding dropout
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),  # Adding dropout
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),  # Adding dropout
            nn.Linear(256, 32)
        )

        # Decoder with Dropout after each ReLU
        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),  # Adding dropout
            nn.Linear(256, 1024),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),  # Adding dropout
            nn.Linear(1024, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),  # Adding dropout
            nn.Linear(4096, 13492),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x