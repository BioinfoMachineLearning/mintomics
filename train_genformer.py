import random
import os
import torch
from torch import optim
from torch import nn
from torch.nn import BCELoss, MSELoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from GenFormer import GenViT as GT  # model
from PrepareDataset import Psedu_data  # Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class FitModel:
    def __init__(self, regression = False, layer = 6, embed_dim = 1024, epoach = 2, batch_size = 25):

        self.epochs = epoach
        self.layer = layer
        self.device = device

        # buid a directory for saving weights
        dir_name = 'Model_Weights'
        self.out_dir = dir_name
        os.makedirs(self.out_dir, exist_ok=True)  # ma

        # load data and get corresponding information
        train_data = Psedu_data(1000)
        _, self.gen_num, self.in_feature = train_data.data.shape  # data shape [Nsample, genlength, feature_dim]
        self.regression = regression

        valid_data = Psedu_data(100)

        # wrap train and valid dataset by DataLoader
        self.train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.valid_data = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

        # load and initialize model
        self.model = GT(
                seq_len = self.gen_num,
                in_dim = self.in_feature,  # input feature
                regression = regression,  # if regression true do regression other do classification
                num_classes = 3,
                dim = embed_dim,   # Last dimension of output tensor after linear transformation
                depth = layer,   # Number of Transformer blocks.
                heads = 8,      # Number of heads in Multi-head Attention layer
                mlp_dim = 2048,     # Dimension of the MLP (FeedForward) layer
                dropout = 0.0,      # Dropout rate.
                emb_dropout = 0.0       # Embedding dropout rate.
            ).to(self.device)

    def fit_model(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Choose Loss function that can be modified to train our model
        if self.regression:
            criterion = MSELoss()
        else:
            criterion = BCELoss()

        best_loss = 10000
        for epoch in range(self.epochs):
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
                # print(out.shape)
                # print(data.shape)
                # print(target.shape)
                loss = criterion(target, out)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_bar.set_description(desc=f"[{epoch}/{self.epochs}] training Loss: {loss:.6f}")

            valid_result = {'nsamples': 0, 'loss': 0, 'acc': 0}
            self.model.eval()
            valid_bar = tqdm(self.valid_data)
            with torch.no_grad():
                for data, target, inf in valid_bar:
                    batch_size = data.shape[0]
                    data = data.to(self.device)
                    target = target.to(self.device)

                    out = self.model(data)
                    out = out[:, inf[0, 0], :]
                    target = torch.transpose(target, 1, 2)
                    target = target[:, inf[0, 1], :]
                    # print(out.shape)
                    # print(data.shape)
                    # print(target.shape)
                    loss = criterion(target, out)

                    valid_result['nsamples'] += batch_size
                    valid_result['loss'] += loss.item() * batch_size
                    valid_bar.set_description(desc=f"[{epoch}/{self.epochs}] validation Loss: {valid_result['loss'] / valid_result['nsamples']:.6f}")

                valid_loss = valid_result['loss'] / valid_result['nsamples']
                now_loss = valid_loss
                if best_loss > now_loss:
                    best_loss = now_loss
                    best_ckpt_file = f'bestg_{self.layer}layers_genfromer.pytorch'
                    # torch.save(self.model.state_dict(), os.path.join(self.out_dir, best_ckpt_file))

        final_ckpt_file = f'finalg_{self.layer}layers_genformer.pytorch'
        # torch.save(self.model.state_dict(), os.path.join(self.out_dir, final_ckpt_file))


if __name__ == '__main__':

    # dataset testing
    '''
    psedu = Psedu_data(3)
    _, Gen_num, feat_dim = psedu.data.shape
    print(psedu.data)
    print(Gen_num, feat_dim)
    '''

    # Training model
    train_model = FitModel(regression=True, epoach = 100)
    train_model.fit_model()