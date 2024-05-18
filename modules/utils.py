import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import KFold
from tqdm import tqdm
import anndata
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import scanpy as sc
from matplotlib.lines import Line2D

writer = SummaryWriter()
test_writer = SummaryWriter()

mapping = {
 '01 IT-ET Glut': 1,
 '02 NP-CT-L6b Glut': 2,
 '03 OB-CR Glut': 3,
 '04 DG-IMN Glut': 4,
 '05 OB-IMN GABA': 5,
 '06 CTX-CGE GABA': 6,
 '07 CTX-MGE GABA': 7,
 '08 CNU-MGE GABA': 8,
 '09 CNU-LGE GABA': 9,
 '10 LSX GABA': 10,
 '11 CNU-HYa GABA': 11,
 '12 HY GABA': 12,
 '13 CNU-HYa Glut': 13,
 '14 HY Glut': 14,
 '15 HY Gnrh1 Glut': 15,
 '16 HY MM Glut': 16,
 '17 MH-LH Glut': 17,
 '18 TH Glut': 18,
 '19 MB Glut': 19,
 '20 MB GABA': 20,
 '21 MB Dopa': 21,
 '22 MB-HB Sero': 22,
 '23 P Glut': 23,
 '24 MY Glut': 24,
 '25 Pineal Glut': 25,
 '26 P GABA': 26,
 '27 MY GABA': 27,
 '28 CB GABA': 28,
 '29 CB Glut': 29,
 '30 Astro-Epen': 30,
 '31 OPC-Oligo': 31,
 '32 OEC': 32,
 '33 Vascular': 33,
 '34 Immune': 34
}

def merge_dataframes(sc_file_path, anno_file_path):
    # Use anndata package to read file
    adata = anndata.read_h5ad(sc_file_path)
    # Normalize the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    # Logarithmize the data
    sc.pp.log1p(adata)
    # Scale the data
    sc.pp.scale(adata, max_value=10)
    # Check if the data is a sparse matrix and convert to dense format
    if isinstance(adata.X, csr_matrix):
        sc_df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
    else:
        sc_df = adata.to_df()
    # Set the index name to 'cell_id'
    sc_df.index.name = 'cell_id'
    # Convert index to string
    sc_df.index = sc_df.index.astype(str)
    # Iterate through each column and remove columns with fewer than 10 non-zero values
    non_zero_counts = sc_df.astype(bool).sum(axis=0)
    sc_df = sc_df.loc[:, non_zero_counts >= 10]
    # Read the file, skipping the first 4 lines
    anno_df = pd.read_csv(anno_file_path, skiprows=4)
    # Set 'cell_id' as the index and keep only the 'class name' column
    anno_df = anno_df.set_index('cell_id')[['class_name']]
    # Convert index to string
    anno_df.index = anno_df.index.astype(str)
    # Convert 'class_name' using the mapping
    anno_df['class_name'] = anno_df['class_name'].map(mapping)
    # Merge dataframes on indexes
    merged_df = sc_df.join(anno_df)
    return merged_df


def build_dataset(*dfs):
    # Ensure there's at least one dataframe
    if not dfs:
        raise ValueError("At least one dataframe must be provided")
    # Find common columns among all dataframes
    common_columns = dfs[0].columns
    for df in dfs[1:]:
        common_columns = common_columns.intersection(df.columns)
    # Select only the common columns from each dataframe
    dfs = [df[common_columns] for df in dfs]
    # Concatenate all dataframes
    result_df = pd.concat(dfs, ignore_index=True)
    return result_df

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

def get_data_splits(X, Y, split, n_splits=5, shuffle=True, random_state=None):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = list(kf.split(X))  
    train_index, test_index = splits[split]  
    
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    return X_train, X_test, Y_train, Y_test


def get_umap(X, Y, tag, mapping):
    mapping = {y:x for x,y in mapping.items()}
    reducer = UMAP(n_neighbors=100, n_components=2)
    embedding = reducer.fit_transform(X)

    plt.figure(figsize=(10, 8))

    unique_targets = np.unique(Y)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_targets)))
    markersize_scatter = 0.1 
    markersize_legend = 10  

    for target, color in zip(unique_targets, colors):
        indices = np.where(Y == target)
        plt.scatter(embedding[indices, 0], embedding[indices, 1], color=color, label=mapping[target], s=markersize_scatter)

    plt.title('UMAP - 2D projection of learned embedding')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=markersize_legend, label=mapping[target])
               for target, color in zip(unique_targets, colors)]
    
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.savefig(f'umap_{tag}.png', bbox_inches='tight')
    plt.show()
