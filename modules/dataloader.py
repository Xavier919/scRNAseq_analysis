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

        for i, sample1 in enumerate(zipped_list):
            available_indices = list(range(len(zipped_list)))
            available_indices.remove(i) 

            for _ in range(self.pairs_per_sample):
                if available_indices:
                    idx = random.choice(available_indices)
                    sample2 = zipped_list[idx]

                    pairs.append((sample1[0], sample2[0]))
                    labels.append(int(sample1[1] == sample2[1]))
                    available_indices.remove(idx)  

        return pairs, torch.tensor(labels, dtype=torch.long)  

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
        self.Y1, self.Y2 = Y
        self.pairs_per_sample = pairs_per_sample
        self.pairs, self.labels1, self.labels2 = self.create_pairs()

    def create_pairs(self):
        zipped_list = list(zip(self.X, self.Y1, self.Y2))
        pairs = []
        labels1 = []
        labels2 = []

        for i, sample1 in enumerate(zipped_list):
            available_indices = list(range(len(zipped_list)))
            available_indices.remove(i)

            for _ in range(self.pairs_per_sample):
                if available_indices:
                    idx = random.choice(available_indices)
                    sample2 = zipped_list[idx]

                    pairs.append((sample1[0], sample2[0]))
                    labels1.append(int(sample1[1] == sample2[1]))
                    labels2.append(int(sample1[2] == sample2[2]))
                    available_indices.remove(idx)

        return pairs, torch.tensor(labels1, dtype=torch.long), torch.tensor(labels2, dtype=torch.long)

    def __len__(self):
        return len(self.labels1)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        sample1 = torch.tensor(pair[0], dtype=torch.float32)
        sample2 = torch.tensor(pair[1], dtype=torch.float32)
        label1 = self.labels1[idx]
        label2 = self.labels2[idx]
        return (sample1, sample2), label1, label2

