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
from modules.kan_model import DeepKAN
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import anndata
import scanpy as sc


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

def merge_dataframes(sc_file_path, anno_file_path):
    # Use anndata package to read file
    adata = anndata.read_h5ad(sc_file_path)

    # Check if the data is a sparse matrix and convert to dataframe
    if isinstance(adata.X, csr_matrix):
        sc_df = pd.DataFrame.sparse.from_spmatrix(adata.X, index=adata.obs_names, columns=adata.var_names)
    else:
        sc_df = adata.to_df()

    # Set the index name to 'cell_id'
    sc_df.index.name = 'cell_id'

    # Convert index to string
    sc_df.index = sc_df.index.astype(str)

    # Drop columns starting with 'mt-'
    sc_df = sc_df.drop(columns=sc_df.filter(like='mt-', axis=1).columns)

    # Iterate through each column and remove columns with fewer than 10 non-zero values
    non_zero_counts = sc_df.astype(bool).sum(axis=0)
    sc_df = sc_df.loc[:, non_zero_counts >= 100]

    # Read the file, skipping the first 4 lines
    anno_df = pd.read_csv(anno_file_path, skiprows=4)

    # Set 'cell_id' as the index and keep only the 'class label' column
    anno_df = anno_df.set_index('cell_id')[['class_label']]

    # Convert index to string
    anno_df.index = anno_df.index.astype(str)

    # Fit and transform the 'class label' column
    anno_df['class_label'] = LabelEncoder().fit_transform(anno_df['class_label'])

    # Merge dataframes on indexes
    merged_df = sc_df.join(anno_df)

    print(merged_df.head())
    return merged_df

writer = SummaryWriter()
test_writer = SummaryWriter()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
    setup(rank, world_size)

    dfA = merge_dataframes('sc_alz/data/A_count.h5ad', 'sc_alz/data/A_mapping.csv')
    dfB = merge_dataframes('sc_alz/data/B_count.h5ad', 'sc_alz/data/B_mapping.csv')
    dfC = merge_dataframes('sc_alz/data/C_count.h5ad', 'sc_alz/data/C_mapping.csv')
    dfD = merge_dataframes('sc_alz/data/D_count.h5ad', 'sc_alz/data/D_mapping.csv')

    merged_df = pd.concat([dfA, dfB, dfC, dfD], ignore_index=True)

    X = merged_df.drop('class_label', axis=1).values

    Y = merged_df['class_label'].values

    X_train, X_test, Y_train, Y_test = get_data_splits(X, Y, args.split, n_splits=5, shuffle=True, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = torch.Tensor(X_train)
    train_dataset = TensorDataset(train_data)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    test_data = torch.Tensor(X_test)
    test_dataset = TensorDataset(test_data)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler)

    #base_net = Autoencoder().to(rank)
    base_net = DeepKAN(X_train.shape[-1], [256,32,256,X_train.shape[-1]]).to(rank)
    base_net = DDP(base_net, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(base_net.parameters(), lr=args.lr)

    best_loss = float('inf')
    no_improvement_count = 0
    early_stopping_limit = 10

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
            torch.save(base_net.module.state_dict(), f'best_auto_{args.split}.pth')
            print(f"Epoch {epoch+1}: Test loss improved, model saved.")
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= early_stopping_limit:
            print("Early stopping triggered.")
            break

    cleanup()
    writer.close()
    test_writer.close()

