import argparse
import torch
from utils import *
from dataloader import PairedDataset
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from mlp_model import MLP, SiameseMLP
from kan_model import DeepKAN, SiameseKAN
import pandas as pd
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("batch_size", type=int)
parser.add_argument("epochs", type=int)
parser.add_argument("lr", type=float)
parser.add_argument("num_pairs", type=int)
parser.add_argument("split", type=int)
parser.add_argument("tag", type=str)
parser.add_argument('-s_layers', nargs="+", type=int)
args = parser.parse_args()


if __name__ == "__main__":

    # Usage
    dfA = merge_dataframes('data/A_count.h5ad', 'data/A_mapping.csv')
    dfB = merge_dataframes('data/B_count.h5ad', 'data/B_mapping.csv')
    dfC = merge_dataframes('data/C_count.h5ad', 'data/C_mapping.csv')
    dfD = merge_dataframes('data/D_count.h5ad', 'data/D_mapping.csv')

    merged_df = build_dataset(dfA, dfB, dfC, dfD)

    # Clear individual dataframes from memory
    del dfA, dfB, dfC, dfD

    X = merged_df.drop(['class_name', 'phenotype'], axis=1).values
    Y = merged_df['class_name'].values
    # Y = merged_df['phenotype'].values

    # Clear merged dataframe from memory
    del merged_df

    X_train, X_test, Y_train, Y_test = get_data_splits(X, Y, args.split, n_splits=5, shuffle=True, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PairedDataset(X_train, Y_train, args.num_pairs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8)

    test_dataset = PairedDataset(X_test, Y_test, args.num_pairs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)


    if args.tag == 'mlp':
        input_dim = X_train.shape[-1]
        shared_layers = list(args.s_layers)
        base_net = MLP(input_dim, shared_layers).to(device)
        siamese_model = SiameseMLP(base_net).to(device)

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
                        noise_scale, base_scale, spline_scale, activation, grid_epsilon, grid_range).to(device)

        siamese_model = SiameseKAN(base_net).to(device)

    optimizer = optim.RMSprop(siamese_model.parameters(), lr=args.lr)

    print(f"Number of pairs: {args.num_pairs}")
    print(f"Batch_size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Split: {args.split}")
    print(f"Tag: {args.tag}")
    print(f"Hidden layers: {args.s_layers}")

    epochs = args.epochs
    best_accuracy = 0
    no_improvement_count = 0 

    for epoch in range(epochs):
        train_loss = train_epoch(siamese_model, train_loader, optimizer, device, epoch, contrastive_loss)
        val_accuracy = eval_model(siamese_model, test_loader, device, epoch, contrastive_loss)
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Accuracy: {val_accuracy}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            no_improvement_count = 0  
            torch.save(siamese_model.base_network.state_dict(), f'{args.tag}_{args.split}_{args.num_pairs}_{list(args.s_layers)[-1]}.pth')
            print("Model saved as best model")
        else:
            no_improvement_count += 1  

        if no_improvement_count >= 5:
            print("No improvement in validation accuracy for 5 consecutive epochs. Training stopped.")
            break