import random
import os
import torch
from torch import optim
from torch import nn
from torch.nn import BCELoss, MSELoss
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr as sprm
from tqdm import tqdm
from GenFormer import GenViT as GT  # model
from PrepareDataset import Psedu_data, Data2target  # Dataset
import wandb  # the logger

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

root = '/bmlfast/joy_RNA/'
class FitModel:
    def __init__(self, regression = False, layer = 1, embed_dim = 128, epoach = 2, batch_size = 64, pertage = 0.15, timepoint = '25n'):

        self.epochs = epoach
        self.layer = layer
        self.device = device
        self.pertage = pertage
        self.mod_name = 'Mlp'
        self.timept = timepoint

        # experiment tracker
        wandb.init(project='joy_RNA')
        wandb.run.name = f'{self.mod_name}_unmask_percentage_:{pertage}_3vs1_timepoint:{timepoint}'
        wandb.run.save()  # get the random run name in my script by Call wandb.run.save(), then get the name with wandb.run.name .

        # buid a directory for saving weights
        dir_name = 'Model_Weights'
        self.out_dir = root+dir_name
        os.makedirs(self.out_dir, exist_ok=True)  # ma

        # load data and get corresponding information
        train_data = Data2target(stage='train', size = 200, pertage = pertage, timepoint=timepoint)  #100 samples each dp
        _, self.gen_num, self.in_feature = train_data.data.shape  # data shape [Nsample, genlength, feature_dim]  maxlize genlengh 16000
        self.regression = regression

        valid_data = Data2target(stage='valid', size = 20, pertage = pertage, timepoint=timepoint)  # 20 samples each dp

        # wrap train and valid dataset by DataLoader
        self.train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.valid_data = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

        # load and initialize model
        self.model = GT(
                seq_len = self.gen_num,
                in_dim = 6,  # input feature
                regression = regression,  # if regression true do regression other do classification
                num_classes = 3,
                dim = embed_dim,   # Last dimension of output tensor after linear transformation
                depth = layer,   # Number of Transformer blocks.
                heads = 1,      # Number of heads in Multi-head Attention layer
                mlp_dim = embed_dim,       # number of Dimension for the MLP (FeedForward) layer
                dropout = 0.0,      # Dropout rate.
                emb_dropout = 0.0,       # Embedding dropout rate.
                model_type='mlp'
            ).to(self.device)
        # print(self.model)
        # exit()

    def fit_model(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        # Choose Loss function that can be modified to train our model
        if self.regression:
            criterion = MSELoss()
        else:
            criterion = BCELoss()

        best_loss = 10000
        best_spcc = -1000
        for epoch in range(self.epochs):
            self.model.train()
            train_bar = tqdm(self.train_data)
            train_loss = 0
            batch_num = 0
            train_spr = 0
            for data, target, inf in train_bar:
                batch_size = data.shape[0]
                # print(f'\n---  the indices: {inf.shape} ----')
                # print(inf[0, 0].shape)
                # print(inf[0, 1].shape)

                data = data.to(self.device)
                data = data[:, :, 0:6]
                target = target.to(self.device)
                # print("------------\n")
                # print(data.shape)
                # print(target.shape)
                # print(inf.shape)

                out = self.model(data)
                # print(out.shape)
                # print(inf[:, 0].shape)
                # out = out[:, inf[:, 0], :]
                out_all = torch.tensor([]).to(self.device)
                target_all = torch.tensor([]).to(self.device)
                for i in range(batch_size):
                    out_n = out[i]
                    target_n = target[i]

                    indice_g = inf[i, 0]
                    indice_p = inf[i, 1]

                    out_nn = out_n[indice_g]
                    target_nn = target_n[indice_p]

                    out_all = torch.cat((out_all, out_nn), 0)
                    target_all  = torch.cat((target_all, target_nn), 0)

                # out = torch.index_select(out, 1, inf[0, 0].to(self.device))
                out = out_all
                target = target_all
                # print(out.shape)
                # target = torch.transpose(target, 1, 2)  # only for mock data, we need transpose it
                # target = target[:, inf[:, 1], :]
                # target = torch.index_select(target, 1, inf[0, 1].to(self.device))
                # print(target.shape)

                loss = criterion(target, out)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                tt = target.flatten().cpu().detach().numpy()
                new = out.flatten().cpu().detach().numpy()
                # pred = torch.isnan(out).any()
                # label = torch.isnan(target).any()
                # print(pred, label)
                # print(f'---- the max of data: {data.max()} and min of data: {data.min()}  max of label: {target.max()} and min of label" {target.min()} ----\n')
                # exit()
                train_bar.set_description(desc=f"[{epoch}/{self.epochs}] training Loss: {loss:.6f} and spcc: {sprm(tt, new)[0]:.6f}")
                train_loss += loss * batch_size
                train_spr += sprm(tt, new)[0] * batch_size
                batch_num += batch_size
            lr_scheduler.step()
                # wandb.log({'train/loss': loss})

            train_loss = train_loss/batch_num
            train_spr = train_spr/batch_num
            valid_result = {'nsamples': 0, 'loss': 0, 'spr': 0}
            self.model.eval()
            valid_bar = tqdm(self.valid_data)
            with torch.no_grad():
                for data, target, inf in valid_bar:
                    batch_size = data.shape[0]
                    data = data.to(self.device)
                    data = data[:, :, 0:6]
                    target = target.to(self.device)

                    out = self.model(data)
                    # out = out[:, inf[:, 0], :]
                    # out = torch.index_select(out, 1, inf[0, 0].to(self.device))
                    # out = out[inf[:, 0], :]
                    # target = torch.transpose(target, 1, 2)   # only for mock data, we need transpose it
                    # target = target[:, inf[:, 1], :]
                    # target = torch.index_select(target, 1, inf[0, 1].to(self.device))
                    # target = target[inf[:, 1], :]

                    # print(out.shape)
                    # print(data.shape)
                    # print(target.shape)

                    out_all = torch.tensor([]).to(self.device)
                    target_all = torch.tensor([]).to(self.device)
                    for i in range(batch_size):
                        out_n = out[i]
                        target_n = target[i]

                        indice_g = inf[i, 0]
                        indice_p = inf[i, 1]

                        out_nn = out_n[indice_g]
                        target_nn = target_n[indice_p]

                        out_all = torch.cat((out_all, out_nn), 0)
                        target_all = torch.cat((target_all, target_nn), 0)

                    target = target_all
                    out = out_all
                    loss = criterion(target, out)

                    tt = target.flatten().cpu().numpy()
                    new = out.flatten().cpu().numpy()


                    valid_result['nsamples'] += batch_size
                    valid_result['loss'] += loss.item() * batch_size
                    valid_result['spr'] += sprm(tt, new)[0] * batch_size

                    valid_bar.set_description(desc=f"[{epoch}/{self.epochs}] validation Loss: {valid_result['loss'] / valid_result['nsamples']:.6f} and spcc: {sprm(tt, new)[0]:.6f}")
                    # wandb.log({'valid/loss': loss})
                    # print(f'the  spearman: {sprm(tt, new)[0]} ---')

                valid_loss = valid_result['loss'] / valid_result['nsamples']
                valid_spr = valid_result['spr'] / valid_result['nsamples']
                now_spcc = valid_spr
                if best_spcc < now_spcc:
                    best_spcc = now_spcc
                    best_ckpt_file = f'bestg_{self.layer}layers_genfromer_{self.pertage}_{self.mod_name}_{self.timept}.pytorch'
                    torch.save(self.model.state_dict(), os.path.join(self.out_dir, best_ckpt_file))

                wandb.log({"Epoch": epoch, 'train/loss': train_loss, 'train/spcc': train_spr, 'valid/loss': valid_loss, 'valid/spcc': valid_spr})

        final_ckpt_file = f'finalg_{self.layer}layers_genformer_{self.pertage}_{self.mod_name}_{self.timept}.pytorch'
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
    train_model = FitModel(regression=True, epoach = 4000, pertage = 0.85)
    train_model.fit_model()