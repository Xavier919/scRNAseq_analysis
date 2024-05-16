import argparse
import torch
from modules.utils import *
from modules.dataloader import PairedDataset
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from modules.mlp_model import MLP, SiameseMLP
from modules.kan_model import DeepKAN, SiameseKAN
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("epochs", type=int)
parser.add_argument("lr", type=float)
parser.add_argument("num_pairs", type=int)
parser.add_argument("split", type=int)
parser.add_argument("tag", type=str)
parser.add_argument('-h_layers', nargs="+", type=int)
args = parser.parse_args()


if __name__ == "__main__":

    #dfA = merge_dataframes('sc_alz/data/A_count.h5ad', 'sc_alz/data/A_mapping.csv')
    #dfB = merge_dataframes('sc_alz/data/B_count.h5ad', 'sc_alz/data/B_mapping.csv')
    #dfC = merge_dataframes('sc_alz/data/C_count.h5ad', 'sc_alz/data/C_mapping.csv')
    #dfD = merge_dataframes('sc_alz/data/D_count.h5ad', 'sc_alz/data/D_mapping.csv')

    merged_df = merge_dataframes('sc_alz/data/fede_count.h5ad', 'sc_alz/data/fede_mapping.csv')

    #merged_df = pd.concat([dfA, dfB, dfC, dfD], ignore_index=True)

    X = merged_df.drop('class_label', axis=1).values

    Y = merged_df['class_label'].values

    X_train, X_test, Y_train, Y_test = get_data_splits(X, Y, args.split, n_splits=5, shuffle=True, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PairedDataset(X_train, Y_train, args.num_pairs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1)

    test_dataset = PairedDataset(X_test, Y_test, args.num_pairs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1)

    hidden_layers = list(args.h_layers)

    if args.tag == 'mlp':
        base_net = MLP(X_train.shape[-1], hidden_layers, output_size=32).to(device)
        siamese_model = SiameseMLP(base_net).to(device)

    elif args.tag == 'kan':
        base_net = DeepKAN(X_train.shape[-1], hidden_layers).to(device)
        siamese_model = SiameseKAN(base_net).to(device)

    optimizer = optim.RMSprop(siamese_model.parameters(), lr=args.lr)

    print(f"Number of pairs: {args.num_pairs}")
    print(f"Batch_size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Split: {args.split}")
    print(f"Tag: {args.tag}")
    print(f"Hidden layers: {args.h_layers}")

    epochs = args.epochs
    best_accuracy = 0
    no_improvement_count = 0 

    for epoch in range(epochs):
        train_loss = train_epoch(siamese_model, train_loader, optimizer, device, epoch)
        val_accuracy = eval_model(siamese_model, test_loader, device, epoch)
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Accuracy: {val_accuracy}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            no_improvement_count = 0  
            torch.save(siamese_model.base_network.state_dict(), f'{args.tag}_{args.split}.pth')
            print("Model saved as best model")
        else:
            no_improvement_count += 1  

        if no_improvement_count >= 10:
            print("No improvement in validation accuracy for 10 consecutive epochs. Training stopped.")
            break