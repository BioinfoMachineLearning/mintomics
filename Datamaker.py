from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import random

import pandas as pd
def create_mock_data(n=10):
    features = []
    labels = []
    mapping = {0: 0, 1: 1, 10: 2, 17: 3, 19: 4, 27: 5, 37: 6, 41: 7, 47: 8, 49: 9}

    for i in range(n):
        k = random.randrange(9)
        genes_2_mask = random.choices(list(mapping.keys()), k=k)
        proteins_2_mask = [mapping[i] for i in genes_2_mask]

        genes_2_mask = torch.tensor(genes_2_mask, dtype=torch.int64)
        proteins_2_mask = torch.tensor(proteins_2_mask, dtype=torch.int64)

        feature = torch.randn(50, 7)
        label = torch.zeros(1, 10)

        feature.index_fill_(0, genes_2_mask, 0)
        label.index_fill_(1, proteins_2_mask, 1)

        features.append(feature)
        labels.append(label)

    features = torch.stack(features, 0)
    labels = torch.stack(labels, 0)
    # labels = torch.transpose(labels, 1, 2)

    return features, labels


class Psedu_data(Dataset):
    def __init__(self, numberSamples = 1000):
        features = []
        labels = []
        #mapping = {0: 0, 1: 1, 10: 2, 17: 3, 19: 4, 27: 5, 37: 6, 41: 7, 47: 8, 49: 9}
        df = pd.read_csv("output.csv",header=None,delimiter="\t")
        mapping = df.set_index(0).to_dict()[1]
        print(mapping)
        info = []
        for _ in range(numberSamples):
            randomNum = random.randrange(12 ,20)
            genes_2_mask = random.choices(list(mapping.keys()), k=randomNum)

            proteins_2_mask = [mapping[i] for i in genes_2_mask]

            genes_2_proten = torch.tensor((genes_2_mask, proteins_2_mask), dtype=torch.int64)

            info.append(genes_2_proten)
            genes_2_mask = torch.tensor(genes_2_mask, dtype=torch.int64)
            proteins_2_mask = torch.tensor(proteins_2_mask, dtype=torch.int64)

            feature = torch.randn(50, 7)
            label = torch.zeros(1, 10)

            feature.index_fill_(0, genes_2_mask, 0)
            label.index_fill_(1, proteins_2_mask, 1)

            features.append(feature)
            labels.append(label)

        features = torch.stack(features, 0)
        labels = torch.stack(labels, 0)

        self.data = features
        self.target = labels
        self.inf = info
        #  print(info)
        # labels = torch.transpose(labels, 1, 2)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return [self.data[idx],  self.target[idx],self.inf[idx]]
    

import torch
from torch.utils.data import Dataset
import random

class CustomDataset(Dataset):
    def __init__(self, items_file, total_samples=1000, mask_range=(0.12, 0.20)):
        self.items = self.read_file(items_file)
        self.items_ind = self.items.index.to_list()
        df = pd.read_csv("output.csv",header=None,delimiter="\t")
        mapping = df.set_index(0).to_dict()[1]
        self.keys = mapping.keys()
        self.total_samples = total_samples
        self.mask_range = mask_range
        self.min_keys_masked_percentage = 0.10

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Randomly determine the percentage of items to mask
        mask_percentage = random.uniform(*self.mask_range)

        # Calculate the number of items and keys to mask
        num_items_to_mask = int(len(self.items_ind) * mask_percentage)
        num_keys_to_mask = max(int(len(self.keys) * mask_percentage), int(len(self.keys) * self.min_keys_masked_percentage))

        # Randomly select items and keys to mask
        masked_items_indices = random.sample(range(len(self.items)), num_items_to_mask)
        masked_keys_indices = random.sample(range(len(self.keys)), num_keys_to_mask)

        # Mask selected items and keys
        masked_items = [self.mask_item(item) if i in masked_items_indices else item for i, item in enumerate(self.items)]
        masked_keys = [self.mask_item(key) if i in masked_keys_indices else key for i, key in enumerate(self.keys)]
        
        proteins_2_mask = [mapping[i] for i in genes_2_mask]

        genes_2_proten = torch.tensor((genes_2_mask, proteins_2_mask), dtype=torch.int64)

        info.append(genes_2_proten)
        return {
            'masked_items': masked_items,
            'masked_keys': masked_keys
        }

    def mask_item(self, item):
        # Implement your item masking logic here
        # For example, replace the item with a special token or apply a specific transformation
        return "MASKED_ITEM"

    def read_file(self, file_path):
        items = pd.read_csv(file_path,index_col=0)
        return items

# Example usage
items_file_path = 'path/to/items.txt'
keys_file_path = 'path/to/keys.txt'
dataset = CustomDataset(items_file_path, keys_file_path, total_samples=1000, mask_range=(0.12, 0.20))

# Access a sample from the dataset
sample = dataset[0]
print(sample)


