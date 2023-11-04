import random

import torch


def create_mock_data(n=10):
    features = []
    labels = []
    mapping = {0: 0, 1: 1, 10: 2, 17: 3, 19: 4, 27: 5, 37: 6, 41: 7, 47: 8, 49: 9}

    for i in range(n):
        k = random.randrange(9)
        genes_2_mask = random.choices(list(mapping.keys()), k=k)
        proteins_2_mask = [mapping[i] for i in genes_2_mask]

        genes_2_mask = torch.tensor(genes_2_mask)
        proteins_2_mask = torch.tensor(proteins_2_mask)

        feature = torch.randn(50, 7)
        label = torch.zeros(1, 10)

        feature.index_fill_(0, genes_2_mask, 0)
        label.index_fill_(1, proteins_2_mask, 1)

        features.append(feature)
        labels.append(label)

    features = torch.stack(features, 0)
    labels = torch.stack(labels, 0)

    return features, labels


if __name__ == '__main__':
    features, labels = create_mock_data(10)

    print(features, labels)
