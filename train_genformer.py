import random
import os
import torch
from torch import optim
from torch import nn
from torch.nn import BCELoss, MSELoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from GenFormer import GenViT as GT  # model
from PrepareDataset import Psedu_data, Data2target  # Dataset
import wandb  # the logger

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

root = '/bmlfast/joy_RNA/'
class FitModel:
    def __init__(self, regression = False, layer = 6, embed_dim = 128, epoach = 2, batch_size = 1, pertage = 0.15):

        self.epochs = epoach
        self.layer = layer
        self.device = device
        self.pertage = pertage

        # experiment tracker
        wandb.init(project='joy_RNA')
        wandb.run.name = f'mask_percentage:{pertage}'
        wandb.run.save()  # get the random run name in my script by Call wandb.run.save(), then get the name with wandb.run.name .

        # buid a directory for saving weights
        dir_name = 'Model_Weights'
        self.out_dir = root+dir_name
        os.makedirs(self.out_dir, exist_ok=True)  # ma

        # load data and get corresponding information
        train_data = Data2target(stage='train', size = 2000, pertage = pertage)  #100 samples each dp
        _, self.gen_num, self.in_feature = train_data.data.shape  # data shape [Nsample, genlength, feature_dim]  maxlize genlengh 16000
        self.regression = regression

        valid_data = Data2target(stage='valid', size = 200, pertage = pertage)  # 20 samples each dp

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
                heads = 4,      # Number of heads in Multi-head Attention layer
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
            train_loss = 0
            batch_num = 0
            for data, target, inf in train_bar:
                batch_size = data.shape[0]
                # print(f'\n---  the indices: {inf.shape} ----')
                # print(inf[0, 0].shape)
                # print(inf[0, 1].shape)

                data = data.to(self.device)
                target = target.to(self.device)
                # print(data.shape)
                # print(target.shape)

                out = self.model(data)
                out = out[:, inf[0, 0], :]
                # target = torch.transpose(target, 1, 2)  # only for mock data, we need transpose it
                target = target[:, inf[0, 1], :]
                # print(out.shape)
                loss = criterion(target, out)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_bar.set_description(desc=f"[{epoch}/{self.epochs}] training Loss: {loss:.6f}")
                train_loss += loss
                batch_num += batch_size

            train_loss = train_loss/batch_num
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
                    # target = torch.transpose(target, 1, 2)   # only for mock data, we need transpose it
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
                    best_ckpt_file = f'bestg_{self.layer}layers_genfromer_{self.pertage}_p.pytorch'
                    torch.save(self.model.state_dict(), os.path.join(self.out_dir, best_ckpt_file))

                wandb.log({"Epoch": epoch, 'train/loss': train_loss, 'valid/loss': valid_loss})

        final_ckpt_file = f'finalg_{self.layer}layers_genformer_{self.pertage}_p.pytorch'
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, final_ckpt_file))


if __name__ == '__main__':

    # dataset testing
    '''
    psedu = Psedu_data(3)
    _, Gen_num, feat_dim = psedu.data.shape
    print(psedu.data)
    print(Gen_num, feat_dim)
    '''

    # Training model
    train_model = FitModel(regression=True, epoach = 5, pertage = 0.15)
    train_model.fit_model()