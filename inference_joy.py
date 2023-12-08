import random
import numpy as np

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import spearmanr as sprm
import os
import torch
from torch import optim
from torch import nn
from torch.nn import BCELoss, MSELoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from GenFormer import GenViT as GT  # model
from PrepareDataset import Psedu_data, Data2target  # Dataset

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

root = '/bmlfast/joy_RNA/'
class FitModel:
    def __init__(self, regression = False, layer = 3, embed_dim = 128, epoach = 2, batch_size = 1, pertage = 0.5, d_percent = 0.5):

        self.epochs = epoach
        self.layer = layer
        self.device = device
        self.pertage = pertage
        self.d_percent =d_percent

        # buid a directory for saving weights
        dir_name = 'Model_Weights'
        self.out_dir = root+dir_name
        os.makedirs(self.out_dir, exist_ok=True)  # ma

        # load data and get corresponding information
        valid_data = Data2target(stage='test', size = 1, pertage = d_percent)  #100 samples each dp
        _, self.gen_num, self.in_feature = valid_data.data.shape  # data shape [Nsample, genlength, feature_dim]  maxlize genlengh 16000
        self.regression = regression

        # wrap valid dataset by DataLoader
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

        best_ckpt_file = f'bestg_{self.layer}layers_genfromer_{self.pertage}_p.pytorch'
        print(best_ckpt_file)
        path = os.path.join(self.out_dir, best_ckpt_file)
        self.model.load_state_dict(torch.load(path))
        # Choose Loss function that can be modified to train our model
        if self.regression:
            criterion = MSELoss()
        else:
            criterion = BCELoss()

        valid_result = {'nsamples': 0, 'loss': 0, 'R2': 0, 'mae': 0, 'spr':0}
        self.model.eval()
        valid_bar = tqdm(self.valid_data)
        with torch.no_grad():
            i = 0
            for data, target, inf in valid_bar:
                i = i + 1
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
                tt = target.flatten().cpu().numpy()
                new = out.flatten().cpu().numpy()
                valid_result['mae'] += mae(tt, new)
                valid_result['R2'] += r2_score(tt, new)
                valid_result['spr'] += sprm(tt, new)[0]
                non_num = np.count_nonzero(new)
                # if non_num != 0:
                print(f'the {i} data length of output: {new.shape} and the number of zeros: {int(new.shape[0])-int(non_num)} and spearman: {sprm(tt, new)[0]} ---')


            valid_loss = valid_result['loss'] / valid_result['nsamples']
            valid_r2 = valid_result['R2'] / valid_result['nsamples']
            valid_mae = valid_result['mae'] / valid_result['nsamples']
            valid_spr = valid_result['spr'] / valid_result['nsamples']

        metrics_file = f'bestg_{self.layer}layers_genfromer_{self.pertage}_data{self.d_percent}.txt'
        pth = root + 'Result'
        file_path = os.path.join(pth, metrics_file)
        record_gds = open(file_path, 'w')
        record_gds.write('\nloss'+":\t" + str(valid_loss))
        record_gds.write('\nr2' + ":\t" + str(valid_r2))
        record_gds.write('\nmae' + ":\t" + str(valid_mae))
        record_gds.write('\nspr' + ":\t" + str(valid_spr))
        record_gds.close()
        print(f'loss: {valid_loss:.6f}  R2: {valid_r2:.6f}  and mae: {valid_mae:.6f} spr: {valid_spr:.6f}\n')

if __name__ == '__main__':

    # dataset testing
    '''
    psedu = Psedu_data(3)
    _, Gen_num, feat_dim = psedu.data.shape
    print(psedu.data)
    print(Gen_num, feat_dim)
    '''

    # Training model
    train_model = FitModel(regression=True, layer = 6, epoach = 100, pertage = 0.15, d_percent=0.0)
    train_model.fit_model()