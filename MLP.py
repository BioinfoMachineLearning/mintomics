import random

import torch
from torch import optim
from torch import nn
from torch.nn import BCELoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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

class MLP_model(nn.Module):
    def __init__(self, in_feature = None, out_feature = None, layer = 3):
        super(MLP_model, self).__init__()  # act_fn = nn.SiLU()

        self.Mlp = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.Linear(in_feature, int(in_feature / 2)),
            nn.Linear(int(in_feature / 2), out_feature),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.Mlp(x)
        # x = x >= 0.5
        # x = x.float()

        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class FitModel:
    def __init__(self, in_feature = 7, out_feature = 1, layer = 3):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.layer = layer
        self.device = device

        # load data to dataloader
        # train_data = create_mock_data(n=10)
        train_data = Psedu_data(1000)
        self.train_data = DataLoader(train_data, batch_size=100, shuffle=True)

        # load model
        self.model = MLP_model(in_feature = self.in_feature, out_feature =self.out_feature, layer =self.layer).to(self.device)

    def fit_model(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = BCELoss()

        for epoch in range(2):
            self.model.train()
            train_bar = tqdm(self.train_data)

            for data, target, inf in train_bar:

                # print(inf.shape)
                # print(inf)
                # print(inf[0, 0])
                # print(inf[0, 1])
                data = data.to(self.device)
                target = target.to(self.device)

                out = self.model(data)
                out = out[:, inf[0, 0], :]
                target = torch.transpose(target, 1, 2)
                target = target[:, inf[0, 1], :]
                print(out.shape)
                # print(data.shape)
                print(target.shape)
                loss = criterion(target, out)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_bar.set_description(desc=f"[{epoch}] training Loss: {loss:.6f}")


if __name__ == '__main__':
    # psedu= create_mock_data(3)
    # psedu = Psedu_data(3)
    # features, labels = create_mock_data(3)

    # print(features, labels)
    '''
    Loader = DataLoader(psedu, batch_size=3, shuffle=True)
    for data, target, inf in Loader:
        print(data.shape)
        print(target.shape)
    '''
    train_model = FitModel(in_feature=7, out_feature=1)
    train_model.fit_model()