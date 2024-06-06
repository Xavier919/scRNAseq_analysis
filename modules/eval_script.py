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
from process_data import *


parser = argparse.ArgumentParser()
parser.add_argument("tag", type=str)
parser.add_argument("model_name", type=str)
parser.add_argument("split", type=int)
parser.add_argument('-s_layers', nargs="+", type=int)

args = parser.parse_args()

if __name__ == "__main__":

    adata1 = anndata.read_h5ad("data/A_count.h5ad")
    adata1.obs['Sample_Tag'] = 1
    adata2 = anndata.read_h5ad("data/B_count.h5ad")
    adata2.obs['Sample_Tag'] = 2
    #adata3 = anndata.read_h5ad("data/C_count.h5ad")
    #adata3.obs['Sample_Tag'] = 3
    #adata4 = anndata.read_h5ad("data/D_count.h5ad")
    #adata4.obs['Sample_Tag'] = 4
    #adata = anndata.concat([adata1, adata2, adata3, adata4], axis=0)
    adata = anndata.concat([adata1, adata2], axis=0)

    #adata = rm_high_mt(adata, threshold=0.3)
    adata = rm_low_exp(adata, threshold=0.05)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    anno_df1 = pd.read_csv("data/A_mapping.csv", skiprows=4)
    anno_df2 = pd.read_csv("data/B_mapping.csv", skiprows=4)
    #anno_df3 = pd.read_csv("data/C_mapping.csv", skiprows=4)
    #anno_df4 = pd.read_csv("data/D_mapping.csv", skiprows=4)

    #anno_df = pd.concat([anno_df1, anno_df2, anno_df3, anno_df4])
    anno_df = pd.concat([anno_df1, anno_df2])
    anno_df = anno_df.set_index('cell_id')[['class_name']]
    anno_df = anno_df['class_name'].map(mapping1)

    sc_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, index=adata.obs_names, columns=adata.var_names)

    sc_df.index = sc_df.index.astype('int64')

    df1 = pd.DataFrame(adata.obs['Sample_Tag'])
    df1.index = df1.index.astype('int64')
    df1 = df1[df1.index.isin(sc_df.index)]

    df2 = pd.DataFrame(anno_df)
    df2.index = df2.index.astype('int64')
    df2 = df2[df2.index.isin(sc_df.index)]


    X = sc_df.values
    Y = df1.values
    #Y = df2.values

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


    #get_umap(outputs, targets, 'umap', mapping1)

    #get_clustering(outputs, Y, 'umap_kmeans', mapping1, n_clusters=4)