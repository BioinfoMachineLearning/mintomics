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
    data = pd.read_csv('Dataset/Data_cpm/Data_control.csv', delimiter=',')
    data.rename(columns={'Unnamed: 0': 'Gene',
                         'IU_Finnerty_Estrus_n1': 'IU_n1',
                         'IU_Finnerty_Estrus_n2': 'IU_n2',
                         'IU_Finnerty_Estrus_n3': 'IU_n3',
                         'IA_Finnerty_Estrus_n1': 'IUA_n1',
                         'IA_Finnerty_Estrus_n2': 'IA_n2',
                         'IA_Finnerty_Estrus_n3': 'IA_n3'}, inplace=True)

    data['Gene'] = data['Gene'].str.upper()
    data.set_index('Gene', inplace=True)
    gene_map = read_map("output_n.csv")
    data['Protein'] = data.index.map(gene_map)

    tf_map = {}
    tf_genes = pd.read_csv('./Mouse_TFs1', header=None)[0].tolist()
    tf_genes = set([i.upper() for i in tf_genes])

    data_genes = set(data.index.tolist())
    for i in data_genes:
        if i in tf_genes:
            tf_map[i] = True
        else:
            tf_map[i] = False

    data['tf'] = data.index.map(tf_map)



    aligned_data = data[(~data['Protein'].isnull())]
    aligned = set(list(aligned_data.index))
    aligned_data = aligned_data[['IU_n1', 'IU_n2', 'IU_n3', 'IUA_n1', 'IA_n2', 'IA_n3']].to_numpy()
    aligned_data = torch.from_numpy(aligned_data).to(torch.float)

    '''tf_data = data[(data['tf'] == True)]
    tf_data = tf_data[['IU_n1', 'IU_n2', 'IU_n3', 'IUA_n1', 'IA_n2', 'IA_n3']].to_numpy()
    tf_data = torch.from_numpy(tf_data).to(torch.float)'''


    all_genes = data[~data.index.isin(aligned)]
    all_genes = all_genes[['IU_n1', 'IU_n2', 'IU_n3', 'IUA_n1', 'IA_n2', 'IA_n3']]

    all_genes = all_genes.to_numpy()
    all_genes = torch.from_numpy(all_genes).to(torch.float)

    return aligned_data, all_genes


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
        self.final = nn.Linear(100, 6)

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


def train_model():
    kwargs = {
        'tf_dimension': 6,
        'embedding_size': 128,
        'aligned_dimension': 6,
        'out_proj': 100
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







train_model()
