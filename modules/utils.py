import pandas as pd
import numpy as np
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import io
import pickle
from sklearn.model_selection import KFold
from tqdm import tqdm
import anndata
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import csr_matrix


writer = SummaryWriter()
test_writer = SummaryWriter()

def build_dataset(df1, df2, df3, df4):
    # Find common columns among all dataframes
    common_columns = df1.columns.intersection(df2.columns).intersection(df3.columns).intersection(df4.columns)
    # Select only the common columns from each dataframe
    df1 = df1[common_columns]
    df2 = df2[common_columns]
    df3 = df3[common_columns]
    df4 = df4[common_columns]
    # Concatenate all dataframes
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    return df

def merge_dataframes(sc_file_path, anno_file_path):
    # Use anndata package to read file
    adata = anndata.read_h5ad(sc_file_path)
    # Check if the data is a sparse matrix and convert to dense format
    if isinstance(adata.X, csr_matrix):
        sc_df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
    else:
        sc_df = adata.to_df()
    # Set the index name to 'cell_id'
    sc_df.index.name = 'cell_id'
    # Convert index to string
    sc_df.index = sc_df.index.astype(str)
    # Drop columns starting with 'mt-'
    sc_df = sc_df.drop(columns=sc_df.filter(like='mt-', axis=1).columns)
    # Iterate through each column and remove columns with fewer than 100 non-zero values
    non_zero_counts = sc_df.astype(bool).sum(axis=0)
    sc_df = sc_df.loc[:, non_zero_counts >= 100]
    # Keep only 1000 samples
    sc_df = sc_df.sample(n=500)
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
    return merged_df

def euclid_dis(vects):
    x, y = vects
    x_flat = x.view(x.size(0), -1)  
    y_flat = y.view(y.size(0), -1)  
    sum_square = torch.sum(torch.square(x_flat - y_flat), axis=1, keepdim=True)
    return torch.sqrt(torch.maximum(sum_square, torch.tensor(torch.finfo(float).eps).to(sum_square.device)))

def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = torch.square(y_pred)
    margin_square = torch.square(torch.clamp(margin - y_pred, min=0))
    loss = torch.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss

def calculate_accuracy(y_pred, y_true):
    pred_labels = (y_pred < 0.5).float()  
    correct = (pred_labels == y_true).float()  
    accuracy = correct.mean()  
    return accuracy

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for (data_a, data_b), target in tqdm(dataloader):
        data_a, data_b, target = data_a.to(device), data_b.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data_a, data_b)  
        loss = contrastive_loss(target, output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        writer.add_scalar("Loss/train", loss.item(), epoch)
    return total_loss / len(dataloader)

def eval_model(model, dataloader, device, epoch):
    model.eval()
    total_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for (data_a, data_b), target in dataloader:
            data_a, data_b, target = data_a.to(device), data_b.to(device), target.to(device)
            output = model(data_a, data_b)
            loss = contrastive_loss(target, output)
            test_writer.add_scalar("Loss/test", loss.item(), epoch)
            accuracy_ = calculate_accuracy(output, target)
            total_accuracy += accuracy_.item() * data_a.size(0)  
            total_samples += data_a.size(0)

    return total_accuracy / total_samples


def evaluate(Y_test, y_pred):
    print("Precision: ",precision_score(Y_test, y_pred, average="weighted", zero_division=0)),
    print("Recall: ", recall_score(Y_test, y_pred, average="weighted", zero_division=0))
    print("F1_score: ", f1_score(Y_test, y_pred, average="weighted", zero_division=0))
    print("accuracy: ", accuracy_score(Y_test, y_pred))

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
        

def get_data_splits(X, Y, split, n_splits=5, shuffle=True, random_state=None):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = list(kf.split(X))  
    train_index, test_index = splits[split]  
    
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    return X_train, X_test, Y_train, Y_test