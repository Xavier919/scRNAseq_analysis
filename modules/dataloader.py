import torch
from torch.utils.data import Dataset
import random

class PairedDataset(Dataset):
    def __init__(self, X, Y, pairs_per_sample):
        self.X = X
        self.Y = Y
        self.pairs_per_sample = pairs_per_sample 
        self.pairs, self.labels = self.create_pairs()

    def create_pairs(self):
        zipped_list = list(zip(self.X, self.Y))
        pairs = []
        labels = []

        for sample1 in zipped_list:
            for _ in range(self.pairs_per_sample):
                sample_choices = [s for s in zipped_list if s != sample1]
                if len(sample_choices) > 0:  
                    sample2 = random.choice(sample_choices)
                    pairs.append((sample1[0], sample2[0]))
                    labels.append(int(sample1[1] == sample2[1]))

        return pairs, torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        sample1 = torch.tensor(pair[0], dtype=torch.float32)
        sample2 = torch.tensor(pair[1], dtype=torch.float32)
        label = self.labels[idx]
        return (sample1, sample2), label
    


class PairedDataset(Dataset):
    def __init__(self, X, Y, pairs_per_sample):
        self.X = X
        self.Y = Y
        self.pairs_per_sample = pairs_per_sample
        self.pairs, self.labels = self.create_pairs()

    def create_pairs(self):
        zipped_list = list(zip(self.X, self.Y))
        pairs = []
        labels = []

        for i, sample1 in enumerate(zipped_list):
            available_indices = list(range(len(zipped_list)))
            available_indices.remove(i)  # Remove current index to avoid self-pairing

            for _ in range(self.pairs_per_sample):
                if available_indices:
                    idx = random.choice(available_indices)
                    sample2 = zipped_list[idx]

                    pairs.append((sample1[0], sample2[0]))
                    labels.append(int(sample1[1] == sample2[1]))
                    available_indices.remove(idx)  # Optional: remove to avoid re-picking

        return pairs, torch.tensor(labels, dtype=torch.long)  # Using torch.long for classification labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        sample1 = torch.tensor(pair[0], dtype=torch.float32)
        sample2 = torch.tensor(pair[1], dtype=torch.float32)
        label = self.labels[idx]
        return (sample1, sample2), label
