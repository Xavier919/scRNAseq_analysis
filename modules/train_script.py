import argparse
import torch
from modules.utils import *
from modules.dataloader import PairedDataset
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from modules.transformer_model import BaseNetTransformer, SiameseTransformer
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("epochs", type=int)
parser.add_argument("lr", type=float)
parser.add_argument("dropout", type=float)
parser.add_argument("num_pairs", type=int)
parser.add_argument("hidden_dim", type=int)
parser.add_argument("num_layers", type=int)
parser.add_argument("num_heads", type=int)
parser.add_argument("split", type=int)
args = parser.parse_args()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
    setup(rank, world_size)


    df1 = sample_cells('human_pancreas_norm.h5ad', 0, num_samples=args.num_samples)
    df2 = sample_cells('Lung_atlas_public.h5ad', 1, num_samples=args.num_samples)

    df = build_dataset(df1, df2)

    X = df.drop('label', axis=1).values
    Y = df['label'].values

    X = X.to_numpy()
    Y = Y.to_numpy()

    X_train, X_test, Y_train, Y_test = get_data_splits(X, Y, args.split, n_splits=5, shuffle=True, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PairedDataset(X_train, Y_train, args.num_pairs)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8)

    test_dataset = PairedDataset(X_test, Y_test, args.num_pairs)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=8)

    base_net = BaseNetTransformer(embedding_dim=1, hidden_dim=args.hidden_dim, num_layers=args.num_layers, n_heads=args.num_heads, dropout=args.dropout)
    siamese_model = SiameseTransformer(base_net).to(rank)
    siamese_model = DDP(siamese_model, device_ids=[rank])

    optimizer = optim.RMSprop(siamese_model.parameters(), lr=args.lr)

    if dist.get_rank() == 0:
        print(f"Number of pairs: {args.num_pairs}")
        print(f"Batch_size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Dropout: {args.dropout}")
        print(f"Epochs: {args.epochs}")
        print(f"Hidden dimensions: {args.hidden_dim}")
        print(f"Number of layers: {args.num_layers}")
        print(f"Number of heads: {args.num_heads}")
        print(f"Split: {args.split}")

    epochs = args.epochs
    best_accuracy = 0
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        train_loss = train_epoch(siamese_model, train_loader, optimizer, device, epoch)
        val_accuracy = eval_model(siamese_model, test_loader, device, epoch)
        if dist.get_rank() == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Accuracy: {val_accuracy}")

        if rank == 0 and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(siamese_model.module.state_dict(), f'best_model_{args.split}.pth')
            torch.save(siamese_model.module.base_network.state_dict(), f'base_net_model_{args.split}.pth')
            print("Model and Base Model saved as best model")

    cleanup()