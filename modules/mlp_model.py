import torch.nn as nn
import torch.nn.functional as F
import torch


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=32):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.dropout = nn.Dropout()
        self.gelu = nn.GELU()
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.gelu(layer(x))  
            #x = self.dropout(x)
        x = self.layers[-1](x) 
        return x
    
class SiameseMLP(nn.Module):
    def __init__(self, base_network):
        super(SiameseMLP, self).__init__()
        self.base_network = base_network
    
    def forward(self, input_a, input_b):
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)
        distance = torch.norm(processed_a - processed_b, p=2, dim=1)
        return distance
    


class MLP(nn.Module):
    def __init__(self, input_size):  
        super(MLP, self).__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 128),
        )
        self.dual = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x1 = self.dual(x)
        x2 = self.dual(x)
        return x1, x2
    
class SiameseMLP(nn.Module):
    def __init__(self, base_network):
        super(SiameseMLP, self).__init__()
        self.base_network = base_network
    
    def forward(self, input_a, input_b):
        type_a, pheno_a = self.base_network(input_a)
        tybe_b, pheno_b = self.base_network(input_b)
        type_distance = torch.norm(type_a - tybe_b, p=2, dim=1)
        pheno_distance = torch.norm(pheno_a - pheno_b, p=2, dim=1)
        return type_distance, pheno_distance