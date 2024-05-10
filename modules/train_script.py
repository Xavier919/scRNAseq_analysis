import argparse
import torch
from modules.utils import *
from modules.dataloader import PairedDataset
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import os
from modules.mlp_model import MLP, SiameseMLP
from modules.kan_model import DeepKAN, SiameseKAN

parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("epochs", type=int)
parser.add_argument("lr", type=float)
parser.add_argument("dropout", type=float)
parser.add_argument("num_pairs", type=int)
parser.add_argument("split", type=int)
args = parser.parse_args()


if __name__ == "__main__":

    df1 = sample_cells('sc_alz/data/human_pancreas_norm.h5ad', 0, num_samples=args.num_samples)
    df2 = sample_cells('sc_alz/data/Lung_atlas_public.h5ad', 1, num_samples=args.num_samples)

    df = build_dataset(df1, df2)

    X = df.drop('label', axis=1).values

    Y = df['label'].values

    X_train, X_test, Y_train, Y_test = get_data_splits(X, Y, args.split, n_splits=5, shuffle=True, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PairedDataset(X_train, Y_train, args.num_pairs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1)

    test_dataset = PairedDataset(X_test, Y_test, args.num_pairs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1)

    base_net = MLP(X_train.shape[-1], [4096,1024,256], output_size=32).to(device)
    siamese_model = SiameseMLP(base_net).to(device)
    
    #base_net = DeepKAN(X_train.shape[-1], [4096,1024,256,32]).to(device)
    #siamese_model = SiameseKAN(base_net).to(device)

    optimizer = optim.RMSprop(siamese_model.parameters(), lr=args.lr)

    print(f"Number of pairs: {args.num_pairs}")
    print(f"Batch_size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Dropout: {args.dropout}")
    print(f"Epochs: {args.epochs}")
    print(f"Split: {args.split}")

    epochs = args.epochs
    best_accuracy = 0
    for epoch in range(epochs):
        train_loss = train_epoch(siamese_model, train_loader, optimizer, device, epoch)
        val_accuracy = eval_model(siamese_model, test_loader, device, epoch)
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Accuracy: {val_accuracy}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(siamese_model.state_dict(), f'siamese_{args.split}.pth')
            torch.save(siamese_model.base_network.state_dict(), f'embed_{args.split}.pth')
            print("Model saved as best model")