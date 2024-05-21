import argparse
import torch
from modules.utils import *
import pickle
import numpy as np
from modules.mlp_model import MLP
from torch.utils.data import DataLoader, TensorDataset
from modules.kan_model import DeepKAN
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
parser.add_argument("tag", type=str)
parser.add_argument("sec_tag", type=str)
parser.add_argument('-h_layers', nargs="+", type=int)

args = parser.parse_args()

if __name__ == "__main__":
    dfA = merge_dataframes('sc_alz/data/A_count.h5ad', 'sc_alz/data/A_mapping.csv')
    dfB = merge_dataframes('sc_alz/data/B_count.h5ad', 'sc_alz/data/B_mapping.csv')
    dfC = merge_dataframes('sc_alz/data/C_count.h5ad', 'sc_alz/data/C_mapping.csv')
    dfD = merge_dataframes('sc_alz/data/D_count.h5ad', 'sc_alz/data/D_mapping.csv')

    merged_df = build_dataset(dfA, dfB, dfC, dfD)

    X = merged_df.drop(['class_name', 'phenotype'], axis=1).values
    Y1 = merged_df['class_name'].values
    Y2 = merged_df['phenotype'].values

    column_names = merged_df.columns.tolist()


    for split in range(4):

        X_train, X_test, Y1_train, Y1_test, Y2_train, Y2_test = get_data_splits(X, Y1, Y2, args.split, n_splits=5, shuffle=True, random_state=42)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_data = torch.Tensor(X_test)
        test_labels1 = torch.Tensor(Y1_test)
        test_labels2 = torch.Tensor(Y2_test)
        test_dataset = TensorDataset(test_data, test_labels1, test_labels2)
        test_loader = DataLoader(test_dataset, batch_size=1)

        hidden_layers = list(args.h_layers)

        if args.tag == 'mlp':
            #base_net = MLP(X_train.shape[-1], hidden_layers, output_size=32)
            base_net = MLP(X_train.shape[-1])

        elif args.tag == 'kan':
            base_net = DeepKAN(X_train.shape[-1], hidden_layers)

        model_path = f'{args.tag}_{split}.pth'
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict = {key: value for key, value in checkpoint.items()}
        base_net.load_state_dict(state_dict)
        base_net.to(device)

        outputs = []
        targets1 = []
        targets2 = []

        base_net.eval()
        torch.no_grad()
        for data_X, Y1, Y2 in test_loader:
            data_X = data_X.to(device)
            output = base_net(data_X)
            outputs.append(output.detach().cpu().numpy()[0])
            targets1.append(int(Y1.detach().numpy()[0]))
            targets2.append(int(Y2.detach().numpy()[0]))

        results = (outputs, targets1, targets2)
        pickle.dump(results, open(f'embed_{args.tag}_{split}.pkl', 'wb'))

        get_umap(np.stack(outputs), targets1, args.tag, args.sec_tag, mapping1)

        get_umap(np.stack(outputs), targets2, args.tag, args.sec_tag, mapping2)