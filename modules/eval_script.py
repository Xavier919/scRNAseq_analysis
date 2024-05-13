import argparse
import torch
from modules.utils import *
import pickle
import numpy as np
from modules.mlp_model import MLP
from torch.utils.data import DataLoader, TensorDataset
from modules.kan_model import DeepKAN

parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
parser.add_argument("model", type=str)
parser.add_argument("tag", type=str)
parser.add_argument('-h_layers', nargs="+", type=int)

args = parser.parse_args()

if __name__ == "__main__":
    df1 = sample_cells('sc_alz/data/human_pancreas_norm.h5ad', 0, num_samples=args.num_samples)
    df2 = sample_cells('sc_alz/data/Lung_atlas_public.h5ad', 1, num_samples=args.num_samples)
    df = build_dataset(df1, df2)
    X = df.drop('label', axis=1).values
    Y = df['label'].values

    for split in range(4):

        X_train, X_test, Y_train, Y_test = get_data_splits(X, Y, split, n_splits=5, shuffle=True, random_state=42)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_data = torch.Tensor(X_test)
        test_labels = torch.Tensor(Y_test)
        test_dataset = TensorDataset(test_data, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=1)

        hidden_layers = list(args.h_layers)

        if args.tag == 'mlp':
            base_net = MLP(X_train.shape[-1], hidden_layers, output_size=32)

        elif args.tag == 'kan':
            base_net = DeepKAN(X_train.shape[-1], hidden_layers)

        model_path = f'{args.tag}_{split}.pth'
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict = {key: value for key, value in checkpoint.items()}
        base_net.load_state_dict(state_dict)
        base_net.to(device)

        outputs = []
        targets = []

        base_net.eval()
        torch.no_grad()
        for data_X, data_Y in test_loader:
            data_X = data_X.to(device)
            output = base_net(data_X)
            outputs.append(output.detach().cpu().numpy()[0])
            targets.append(int(data_Y.numpy()[0]))

        results = (outputs, targets)
        pickle.dump(results, open(f'embed_{args.tag}_{split}.pkl', 'wb'))