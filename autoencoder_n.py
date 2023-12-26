import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric import nn
import torch.nn
from torch import nn
import torch.nn.functional as F
from scipy.stats.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score


def read_map(filename, header=False):
    map = {}
    with open(filename) as f:
        lines = f.readlines()
        if header:
            lines = lines[1:]
        for line in lines:
            spli = line.strip("\n").split("\t")
            map[spli[0].upper()] = spli[1]
    return map


def create_data():
    Gene_file = ['Data_control.csv', 'Data_0_5preg.csv', 'Data_1_5preg.csv', 'Data_2_5preg.csv']
    Pro_file = ['Labels_control.csv', 'Labels_0_5preg.csv', 'Labels_1_5preg.csv', 'Labels_2_5preg.csv']

    data = pd.read_csv('Dataset/Data_cpm/Data_0_5preg.csv', delimiter=',')
    print(data.columns)
    data.rename(columns={'Unnamed: 0': 'Gene',
                         'IU_0.5preg_n1': 'IU_n1',
                         'IU_0.5preg_n2': 'IU_n2',
                         'IU_0.5preg_n3': 'IU_n3',
                         'IA_0.5preg_n1': 'IUA_n1',
                         'IA_0.5preg_n2': 'IA_n2',
                         'IA_0.5preg_n3': 'IA_n3'}, inplace=True)

    data['Gene'] = data['Gene'].str.upper()
    data.set_index('Gene', inplace=True)
    gene_map = read_map("output_n.csv")
    data['Protein'] = data.index.map(gene_map)

    tf_map = {}
    tf_genes = pd.read_csv('./Mouse_TFs1.csv', header=None)[0].tolist()
    tf_genes = set([i.upper() for i in tf_genes])

    data_genes = set(data.index.tolist())
    for i in data_genes:
        if i in tf_genes:
            tf_map[i] = True
        else:
            tf_map[i] = False

    data['tf'] = data.index.map(tf_map)   # add tf information in it
    # print(data)



    aligned_data = data[(~data['Protein'].isnull())]

    aligned = set(list(aligned_data.index))
    # print(aligned)
    aligned_data = aligned_data[['IU_n1', 'IU_n2', 'IU_n3', 'IUA_n1', 'IA_n2', 'IA_n3']].to_numpy()
    # aligned_data = torch.from_numpy(aligned_data).to(torch.float)
    per_a = np.percentile(aligned_data, 99.9)
    aligned_data = np.clip(aligned_data, 0.0, per_a)
    aligned_data = aligned_data / per_a
    aligned_data = torch.from_numpy(aligned_data).to(torch.float)

    '''tf_data = data[(data['tf'] == True)]  # for tf data 
    tf_data = tf_data[['IU_n1', 'IU_n2', 'IU_n3', 'IUA_n1', 'IA_n2', 'IA_n3']].to_numpy()
    tf_data = torch.from_numpy(tf_data).to(torch.float)'''

    all_genes = data[~data.index.isin(aligned)]
    # print(all_genes)

    all_genes = data[['IU_n1', 'IU_n2', 'IU_n3', 'IUA_n1', 'IA_n2', 'IA_n3']]

    all_genes = all_genes.to_numpy()
    # normalize the gene datasets as following steps
    per_a = np.percentile(all_genes, 99.9)
    all_genes = np.clip(all_genes, 0.0, per_a)
    all_genes = all_genes / per_a
    all_genes = torch.from_numpy(all_genes).to(torch.float)
    print(f'----- the geneID data shape: {all_genes.shape} -----\n')

    Prodir = './Dataset/Labels_proc_log10_minmax/'
    Pro_file = ['Labels_0_5preg.csv']

    all_proteins = None
    for m, n in enumerate(Pro_file):
        print(n)
        Prot = pd.read_csv(Prodir + n)
        Prot_dat = Prot[Prot.columns[1:2]].values.astype(np.float32)
        all_proteins = torch.from_numpy(Prot_dat).to(torch.float)
    print(f'----- the protein data shape: {all_proteins.shape} -----\n')


    return all_genes, all_proteins


class SelfAttention(nn.Module):

    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__()

        # tf_dimension = kwargs['tf_dimension']
        embedding_size = kwargs['embedding_size']
        aligned_dimension = kwargs['aligned_dimension']
        # out_proj = kwargs['out_proj']
        # self.tf_genes = kwargs['tf_genes']

        # GO terms as keys & values
        self.q = nn.Linear(aligned_dimension, embedding_size)
        self.k = nn.Linear(aligned_dimension, embedding_size)
        self.v = nn.Linear(aligned_dimension, embedding_size)
        # self.final = nn.Linear(embedding_size, out_proj)

    def forward(self, x_1):
        q = self.q(x_1)
        k = self.k(x_1)
        v = self.v(x_1)

        dk = q.size()[-1]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(dk)

        attention = F.softmax(scores, dim=-1)
        values = torch.matmul(attention, v)

        # sum_out = values + q
        sum_out = q
        # proj_out = self.final(sum_out)
        return sum_out  #, attention

class CrossAttentionB(nn.Module):

    def __init__(self, **kwargs):
        super(CrossAttentionB, self).__init__()

        # tf_dimension = kwargs['tf_dimension']
        embedding_size = kwargs['embedding_size']
        aligned_dimension = kwargs['protein_dimension']
        out_proj = kwargs['out_proj']
        # self.tf_genes = kwargs['tf_genes']

        # proteins as queries
        self.q = nn.Linear(aligned_dimension, embedding_size)
        # GO terms as keys & values
        # self.k = nn.Linear(tf_dimension, embedding_size)
        # self.v = nn.Linear(tf_dimension, embedding_size)
        self.final = nn.Linear(embedding_size, out_proj)

    def forward(self, x_1, en_out):
        q = self.q(x_1)
        # k = self.k(self.tf_genes)
        # v = self.v(self.tf_genes)
        k = en_out
        v = en_out

        dk = q.size()[-1]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(dk)

        attention = F.softmax(scores, dim=-1)
        values = torch.matmul(attention, v)

        sum_out = values + q
        proj_out = self.final(sum_out)
        return proj_out, attention

class CrossAttention(nn.Module):

    def __init__(self, **kwargs):
        super(CrossAttention, self).__init__()

        tf_dimension = kwargs['tf_dimension']
        embedding_size = kwargs['embedding_size']
        aligned_dimension = kwargs['aligned_dimension']
        out_proj = kwargs['out_proj']
        self.tf_genes = kwargs['tf_genes']

        # proteins as queries
        self.q = nn.Linear(aligned_dimension, embedding_size)
        # GO terms as keys & values
        self.k = nn.Linear(tf_dimension, embedding_size)
        self.v = nn.Linear(tf_dimension, embedding_size)
        self.final = nn.Linear(embedding_size, out_proj)

    def forward(self, x_1):
        q = self.q(x_1)
        k = self.k(self.tf_genes)
        v = self.k(self.tf_genes)

        dk = q.size()[-1]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(dk)

        attention = F.softmax(scores, dim=-1)
        values = torch.matmul(attention, v)

        sum_out = values + q
        proj_out = self.final(sum_out)
        return proj_out, attention


class ReverseAttention(nn.Module):

    def __init__(self, **kwargs):
        super(ReverseAttention, self).__init__()
        in_dimension = kwargs['out_proj']
        out_dimension = kwargs['final_out']
        self.final = nn.Sequential(# nn.ReLU(),
                                   nn.Linear(in_dimension, out_dimension)
                                   )

    def forward(self, x):
        x = self.final(x)
        return x


class AE(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = CrossAttention(**kwargs)
        self.decoder = ReverseAttention(**kwargs)

    def forward(self, x):
        encoded, attention = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, attention

class Transformer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = SelfAttention(**kwargs)
        self.decoder = CrossAttentionB(**kwargs)
        self.final = ReverseAttention(**kwargs)

    def forward(self, x, y):
        # encoded, encode_attention = self.encoder(x)
        encoded = self.encoder(x)
        decoded, decode_attention = self.decoder(y, encoded)
        decoded = self.final(decoded)
        return decoded, decode_attention


def showAttention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    # ax.set_xticklabels([''] + input_sentence.split(' ') +
    #                   ['<EOS>'], rotation=90)
    # ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def train_model():
    kwargs = {
        'tf_dimension': 6,
        'embedding_size': 128,
        'aligned_dimension': 6,
        'out_proj': 100,
        'final_out': 6
    }

    aligned_genes, tf_genes = create_data()
    tgts = torch.zeros(aligned_genes.shape[0])

    my_dataset = TensorDataset(aligned_genes, tgts)
    my_dataloader = DataLoader(dataset=my_dataset,
                               shuffle=False,
                               batch_size=3000)
    kwargs['tf_genes'] = tf_genes

    model = AE(**kwargs)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001,
                                 weight_decay=1e-8)
    cos_col = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_row = nn.CosineSimilarity(dim=0, eps=1e-6)

    epochs = 10
    losses = []
    avg_cos_row = []
    avg_cos_col = []
    for epoch in range(epochs):
        print(epoch)
        for (feature, _) in my_dataloader:

            reconstructed, attention = model(feature)

            # Calculating the loss function
            loss = loss_function(reconstructed, feature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # r2 = np.corrcoef(feature.detach().numpy(), feature.detach().numpy())
            # pc = pearsonr(feature.detach().numpy().tolist(), reconstructed.detach().numpy().tolist())

            cos_sim_row = cos_row(reconstructed, feature).mean()
            cos_sim_col = cos_col(reconstructed, feature).mean()

            # print(roc_auc_score(feature.detach().numpy(), feature.detach().numpy()))

            #print(average_precision_score(feature.detach().numpy(), feature.detach().numpy()))

            # Storing the losses in a list for plotting
            losses.append(loss.detach().numpy())
            avg_cos_row.append(cos_sim_row.detach().numpy())
            avg_cos_col.append(cos_sim_col.detach().numpy())

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax[0].plot(losses)
    ax[0].set_title('Losses')

    ax[1].plot(avg_cos_row)
    ax[1].set_title('Rowise Cosine Similarity')

    ax[2].plot(avg_cos_col)
    ax[2].set_title('Columnwise Cosine Similarity')

    fig.suptitle('Results')
    plt.show()

    showAttention(None, None, attention.detach())

    att = (attention > 0.1)# .float()

    print(attention[att].shape)

    print(torch.sum(att, dim=0))

from scipy.stats import spearmanr as sprm
def train_TFmodel():
    kwargs = {
        'tf_dimension': 6,
        'protein_dimension': 1,
        'embedding_size': 128,
        'aligned_dimension': 6,
        'out_proj': 100,
        'final_out': 1
    }

    '''
    aligned_genes, tf_genes = create_data()
    tgts = torch.zeros(aligned_genes.shape[0])
    my_dataset = TensorDataset(aligned_genes, tgts)
    '''

    data, target = create_data()

    '''
    my_dataset = indata(data)
    my_dataloader = DataLoader(dataset=my_dataset,
                               shuffle=False,
                               batch_size=3000)
    '''

    model = Transformer(**kwargs).to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001,
                                 weight_decay=1e-8)
    cos_col = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_row = nn.CosineSimilarity(dim=0, eps=1e-6)

    epochs = 100
    losses = []
    avg_cos_row = []
    avg_cos_col = []
    spr_all = []
    attention = None
    cross_attention = None

    for epoch in range(epochs):
            print(epoch)
            model.train()
            feature, target = create_data()

            reconstructed, cross_attention = model(feature.to(device), target.to(device))

            # Calculating the loss function
            loss = loss_function(reconstructed, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # r2 = np.corrcoef(feature.detach().numpy(), feature.detach().numpy())
            # pc = pearsonr(feature.detach().numpy().tolist(), reconstructed.detach().numpy().tolist())

            cos_sim_row = cos_row(reconstructed.cpu(), target).mean()
            cos_sim_col = cos_col(reconstructed.cpu(), target).mean()
            spr_epoch = sprm(reconstructed.cpu().detach().numpy(), target.detach().numpy())[0]
            spr_all.append(spr_epoch)

            # print(roc_auc_score(feature.detach().numpy(), feature.detach().numpy()))

            #print(average_precision_score(feature.detach().numpy(), feature.detach().numpy()))

            # Storing the losses in a list for plotting
            losses.append(loss.cpu().detach().numpy())
            avg_cos_row.append(cos_sim_row.detach().numpy())
            avg_cos_col.append(cos_sim_col.detach().numpy())

    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    ax[0].plot(losses)
    ax[0].set_title('Losses')

    ax[1].plot(avg_cos_row)
    ax[1].set_title('Rowise Cosine Similarity')

    ax[2].plot(avg_cos_col)
    ax[2].set_title('Columnwise Cosine Similarity')

    ax[3].plot(spr_all)
    ax[3].set_title('spearman vs epoch')

    fig.suptitle('Results')
    plt.show()

    # showAttention(None, None, attention.detach())
    showAttention(None, None, cross_attention.detach())

    att = (cross_attention > 0.01)# .float()

    print(cross_attention[att].shape)
    print(f'----- the max attention score: {cross_attention.max()} -----')

    print(torch.sum(att, dim=0))
    print(spr_all[-1])


if __name__ == "__main__":
    train_TFmodel()
    # create_data()
