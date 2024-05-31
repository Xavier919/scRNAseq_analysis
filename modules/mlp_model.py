import torch.nn as nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    def __init__(self, input_size, encoder_layers, dual_layers, activation=nn.GELU):
        super(MLP, self).__init__()
        self.encoder = self._build_layers(input_size, encoder_layers, activation)
        self.dual1 = self._build_layers(encoder_layers[-1], dual_layers, activation)
        self.dual2 = self._build_layers(encoder_layers[-1], dual_layers, activation)

    def _build_layers(self, input_dim, layer_dims, activation):
        layers = []
        for output_dim in layer_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(activation())
            input_dim = output_dim
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.encoder(x)
        x1 = self.dual1(x)
        x2 = self.dual2(x)
        return x1, x2

class SiameseMLP(nn.Module):
    def __init__(self, base_network):
        super(SiameseMLP, self).__init__()
        self.base_network = base_network
    
    def forward(self, input_a, input_b):
        type_a, pheno_a = self.base_network(input_a)
        type_b, pheno_b = self.base_network(input_b)
        type_distance = torch.norm(type_a - type_b, p=2, dim=1)
        pheno_distance = torch.norm(pheno_a - pheno_b, p=2, dim=1)
        return type_distance, pheno_distance
    


class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        
        # Create the first layer with input_size
        self.layers.append(nn.Linear(input_size, layer_sizes[0]))
        self.batchnorms.append(nn.BatchNorm1d(layer_sizes[0]))
        
        # Create subsequent layers based on layer_sizes
        for i in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            self.batchnorms.append(nn.BatchNorm1d(layer_sizes[i]))
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.batchnorms[i](x)
            x = self.activation(x)
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
        #distance = torch.sum((processed_a - processed_b) ** 2, dim=1)
        return distance