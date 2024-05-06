import argparse
import torch
from modules.utils import *
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import os
from modules.autoencoder_model import Autoencoder
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("epochs", type=int)
parser.add_argument("lr", type=float)
parser.add_argument("split", type=int)
args = parser.parse_args()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

writer = SummaryWriter()
test_writer = SummaryWriter()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
    setup(rank, world_size)

    # Data preparation
    df1 = sample_cells('sc_alz/data/human_pancreas_norm.h5ad', 0, num_samples=args.num_samples)
    df2 = sample_cells('sc_alz/data/Lung_atlas_public.h5ad', 1, num_samples=args.num_samples)
    df = build_dataset(df1, df2)
    X = df.drop('label', axis=1).values
    Y = df['label'].values

    # Splitting data
    X_train, X_test, Y_train, Y_test = get_data_splits(X, Y, args.split, n_splits=5, shuffle=True, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setting up data loaders for distributed training
    train_data = torch.Tensor(X_train)
    train_dataset = TensorDataset(train_data)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    test_data = torch.Tensor(X_test)
    test_dataset = TensorDataset(test_data)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler)

    # Model setup
    base_net = Autoencoder().to(rank)
    base_net = DDP(base_net, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(base_net.parameters(), lr=args.lr)

    # Initialize logging and early stopping parameters
    best_loss = float('inf')
    no_improvement_count = 0
    early_stopping_limit = 5

    writer = SummaryWriter()
    test_writer = SummaryWriter()

    for epoch in range(args.epochs):
        base_net.train()
        total_loss = 0
        num_batches = 0
        
        for data in train_loader:
            img, = data
            img = img.to(device)
            output = base_net(img)
            loss = criterion(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        mean_train_loss = total_loss / num_batches
        print('Epoch [{}/{}], Train loss: {:.4f}'.format(epoch+1, args.epochs, mean_train_loss))
        writer.add_scalar("Loss/train", mean_train_loss, epoch)

        # Evaluate on test data
        base_net.eval()
        test_loss = 0
        num_batches = 0
        with torch.no_grad():
            for data in test_loader:
                img, = data
                img = img.to(device)
                output = base_net(img)
                loss = criterion(output, img)
                test_loss += loss.item()
                num_batches += 1
        mean_test_loss = test_loss / num_batches
        print('Epoch [{}/{}], Test loss: {:.4f}'.format(epoch+1, args.epochs, mean_test_loss))
        test_writer.add_scalar("Loss/test", mean_test_loss, epoch)

        if mean_test_loss < best_loss:
            best_loss = mean_test_loss
            no_improvement_count = 0
            torch.save(base_net.module.state_dict(), 'best_auto.pth')
            print(f"Epoch {epoch+1}: Test loss improved, model saved.")
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= early_stopping_limit:
            print("Early stopping triggered.")
            break

    cleanup()
    writer.close()
    test_writer.close()

