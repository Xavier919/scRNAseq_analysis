import argparse
import torch
from utils import *
import pickle
import numpy as np
from mlp_model import MLP
from torch.utils.data import DataLoader, TensorDataset
from kan_model import DeepKAN
import torch.nn as nn
from tqdm import tqdm
import anndata


parser = argparse.ArgumentParser()
parser.add_argument("tag", type=str)
parser.add_argument("model_name", type=str)
parser.add_argument("split", type=int)
parser.add_argument('-s_layers', nargs="+", type=int)

args = parser.parse_args()

if __name__ == "__main__":

    file_path = 'data/filtered_adata.h5ad'
    adata = anndata.read_h5ad(file_path)

    Y = np.array(adata.obs['target'].tolist())
    X = adata.X.toarray()

    X_train, X_test, Y_train, Y_test = get_data_splits(X, Y, args.split, n_splits=5, shuffle=True, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = torch.Tensor(X_test)
    test_labels = torch.Tensor(Y_test)
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8)

    if args.tag == 'mlp':
        input_dim = X_train.shape[-1]
        shared_layers = list(args.s_layers)
        base_net = MLP(input_dim, shared_layers)

    elif args.tag == 'kan':
        input_dim = X_train.shape[-1]
        shared_layers = list(args.s_layers)
        num_knots = 5
        spline_order = 3
        noise_scale = 0.1
        base_scale = 1.0
        spline_scale = 1.0
        activation = nn.SiLU
        grid_epsilon = 0.02
        grid_range = [-1, 1]
        
        base_net = DeepKAN(input_dim, shared_layers, num_knots, spline_order,
                        noise_scale, base_scale, spline_scale, activation, grid_epsilon, grid_range)

    model_path = args.model_name
    checkpoint = torch.load(model_path, map_location=device)
    base_net.load_state_dict(checkpoint)
    base_net.to(device)

    outputs = []
    targets = []

    base_net.eval()
    with torch.no_grad():
        for data_X, Y in tqdm(test_loader):
            data_X = data_X.to(device)
            output = base_net(data_X.view(1, -1))
            outputs.append(output.detach().cpu().numpy()[0])
            targets.append(int(Y.detach().cpu().numpy()[0]))
    outputs = np.stack(outputs)
    results = (outputs, targets)
    with open(f'embed_{args.model_name}.pkl', 'wb') as f:
        pickle.dump(results, f)

