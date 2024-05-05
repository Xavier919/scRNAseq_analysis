import argparse
import torch
from modules.preprocess import text_edit
from modules.utils import *
from gensim.models import KeyedVectors
from modules.transformer_model import BaseNetTransformer
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
parser.add_argument("hidden_dim", type=int)
parser.add_argument("num_layers", type=int)
parser.add_argument("num_heads", type=int)
parser.add_argument("split", type=int)
args = parser.parse_args()


if __name__ == "__main__":

    dataset = build_dataset(path="siamese_net/data",num_samples=args.num_samples, rnd_state=10)

    dataset = text_edit(dataset, grp_num=False, rm_newline=True, rm_punctuation=True, lowercase=True, lemmatize=False, html_=True, expand=False)

    X = np.array([x['text'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'international', 'arts', 'affaires']])
    Y = np.array([x['section_label'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'international', 'arts', 'affaires']])

    X_train, X_test, Y_train, Y_test = get_data_splits(X, Y, args.split, n_splits=5, shuffle=True, random_state=42)

    model_path = 'wiki.fr.vec'
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    base_net = BaseNetTransformer(embedding_dim=300, hidden_dim=args.hidden_dim, num_layers=args.num_layers, n_heads=args.num_heads, out_features=32)

    model_path = f"base_net_model_{args.split}.pth"
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    base_net.load_state_dict(state_dict)

    base_net = base_net.to(device)

    base_net.eval()
    results = []
    for X_, Y_ in  list(zip(X_test,Y_test)):
        X_ = torch.tensor(X_).view(1,1,-1).float().to(device)
        output = base_net(X_).detach()
        results.append((output, Y_))

    pickle.dump(results, open(f'results_{args.split}.pkl', 'wb'))