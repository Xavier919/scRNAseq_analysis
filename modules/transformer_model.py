import torch
import torch.nn as nn

class BaseNetTransformer(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, num_layers=1, n_heads=1, dropout=0.1, out_features=32):
        super(BaseNetTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout, activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, out_features)

    def forward(self, x):
        x = x.transpose(1, 2)  
        transformer_out = self.transformer_encoder(x)
        out = transformer_out.mean(dim=1)
        out = self.fc(out)
        return out

class SiameseTransformer(nn.Module):
    def __init__(self, base_network):
        super(SiameseTransformer, self).__init__()
        self.base_network = base_network
    
    def forward(self, input_a, input_b):
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)
        distance = torch.norm(processed_a - processed_b, p=2, dim=1)
        return distance