import argparse
import torch
from modules.utils import *
import pickle
import numpy as np
from modules.mlp_model import MLP
from torch.utils.data import DataLoader, TensorDataset


parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
parser.add_argument("model", type=str)
parser.add_argument("split", type=int)
args = parser.parse_args()

if __name__ == "__main__":
    # Data preparation
    df1 = sample_cells('sc_alz/data/human_pancreas_norm.h5ad', 0, num_samples=args.num_samples)
    df2 = sample_cells('sc_alz/data/Lung_atlas_public.h5ad', 1, num_samples=args.num_samples)
    df = build_dataset(df1, df2)
    X = df.drop('label', axis=1).values
    Y = df['label'].values

    # Splitting data
    X_train, X_test, Y_train, Y_test = get_data_splits(X, Y, args.split, n_splits=5, shuffle=True, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = torch.Tensor(X_test)
    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1)

    base_net = MLP(X_test.shape[-1], [4096,1024,256], output_size=32)
    model_path = args.model
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    base_net.load_state_dict(state_dict)

    predictions = []
    base_net.eval()
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0]
            data = data.to(device)
            output = base_net(data)
            predictions.append(output.detach().cpu().numpy())
    predictions = np.stack(predictions)

    pickle.dump(predictions, open(f'predictions_{args.split}.pkl', 'wb'))