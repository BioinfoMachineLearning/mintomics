import random
import torch
from torch.utils.data import Dataset


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
    def __init__(self, n = 10):
        features = []
        labels = []
        mapping = {0: 0, 1: 1, 10: 2, 17: 3, 19: 4, 27: 5, 37: 6, 41: 7, 47: 8, 49: 9}

        info = []
        for i in range(n):
            k = random.randrange(9)
            genes_2_mask = random.choices(list(mapping.keys()), k=4)
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
        return self.data[idx],  self.target[idx],self.inf[idx]


def logit_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate how much classes in logits were correctly
    predicted.

    Args:
        logits: output of the model
        target: target class indices

    Returns:

    """
    idx = logits.max(1).indices
    acc = (idx == target).int()
    return acc.sum() / torch.numel(acc)