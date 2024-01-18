import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
DATA_DIR = '/home/aghktb/JOYS_Project/mintomics'
root = '/bmlfast/joy_RNA/Data/'
dir_in = 'bulkRNA_p_'
dir_out = 'protein_p_'
dir_index = 'mapping_p_'

# Plot data to find out the problems
def plotTime(data):
    print(data.shape)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    x = [i+1 for i in range(data[0].size(0))]
    y_0_5 = data[0].squeeze(1)
    y_1_5 = data[1].squeeze(1) # +1.02
    y_2_5 = data[2].squeeze(1) # +2.02
    y_3_5 = data[3].squeeze(1) # +3.02

    per_5 = torch.quantile(y_0_5, 0.95)
    non_zero = y_0_5[y_0_5 > per_5].mean()
    print(f'-------- the mean of nonzero is: {non_zero}----------------')

    ax.plot(x,  y_0_5, color='crimson', label='0.5')
    #ax.plot(x, y_1_5, color='green', label='1.5')
    #ax.plot(x, y_2_5, color='blue', label='2.5')
    #ax.plot(x, y_3_5, color='violet', label='Control')
    ax.set_ylabel("abundance", fontsize=14)
    ax.set_xlabel("Indices", fontsize=14)
    ax.set_title("Protein/Time", fontsize=14)
    plt.legend(loc='upper right')
    plt.show()

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


def gene2protein(stage = 'train', size = 10, pertage = 0.5):
    features = []
    labels = []

    # load transcription factors
    TF = pd.read_csv(DATA_DIR+'/Mouse_TFs1', header=None)  # without header, the column index will be 0
    TF = TF[0].tolist()
    TF = [i.upper() for i in TF]
    TF = set(TF)
    print(len(TF))
    # print(data[0].tolist())

    # load data
    Genedir = DATA_DIR+'/Dataset/Data_cpm/'
    Prodir = DATA_DIR+'/Dataset/Labels_proc_log10_minmax/'
    Gene_file = [ 'Data_2_5preg.csv']
    Pro_file = ['Labels_2_5preg.csv']


    # load geneID data, information and add TF feature to the original dataset
    TF_label = None
    GeneID_t = None
    Gene_all = []

    for m, n in enumerate(Gene_file):
        print(n)
        Gene = pd.read_csv(Genedir + n)
        # Gene = Gene.head(16000)
        if m == 0:
            # print(Gene['Unnamed: 0'])
            GeneID = Gene['Unnamed: 0'].tolist()
            GeneID = [i.upper() for i in GeneID]
            GeneID_t = GeneID
            GeneID = set(GeneID)

            Intersect = GeneID.intersection(TF)
            # print(len(Intersect))

            TF_label = [0.0 for i in range(len(GeneID))]
            # print(len(TF_label))

            # assign the value 1.0 to the TF geneID
            for (i, j) in enumerate(GeneID):
                if j in Intersect:
                    TF_label[i] = 1.0

        print(f'{m} and number of TF: {TF_label.count(1.0)}\n')
        Gene['TF'] = TF_label
        # convert dataframe to matrix
        Gene_dat = Gene[Gene.columns[1:8]].values.astype(np.float32)

        # Normailize data by the difference between max and min
        # min = Gene_dat.min()
        # max = Gene_dat.max()
        # Gene_dat = (Gene_dat - min) / (max - min)
        per_a = np.percentile(Gene_dat, 99.9)
        Gene_dat = np.clip(Gene_dat, 0.0, per_a)
        Gene_dat = Gene_dat / per_a

        Gene_all.append(Gene_dat)
        print(Gene_dat.shape)

    
    # load protein data and information
    Prot_ID = None
    Prot_all = []
    for m, n in enumerate(Pro_file):
        
        Prot = pd.read_csv(Prodir + n)
        if m == 0:
            Prot_ID = Prot['Accession'].tolist()
            Prot_ID = [i.upper() for i in Prot_ID]
        Prot_dat = Prot[Prot.columns[1:2]].values.astype(np.float32)

        # Normailize data by the difference between max and min
        # min = Prot_dat.min()
        # max = Prot_dat.max()
        # Prot_dat = (Prot_dat-min)/(max-min)
        #per_a = np.percentile(Prot_dat, 99.9)
        #Prot_dat = np.clip(Prot_dat, 0.0, per_a)
        #Prot_dat = Prot_dat/per_a

        print(f'find out the min value: {Prot_dat.min()} and max value: {Prot_dat.max()}\n')
        Prot_all.append(Prot_dat)
        # print(Prot_dat.shape)


    # load the mapping between GeneID and Protein accession, and convert the mapping to index pair
    mapping = pd.read_csv(DATA_DIR+'/output_n.csv', delimiter='\t', header=None)
    # print(mapping)
    Gene_map = mapping[mapping.columns[0]].tolist()
    Gene_map = [i.upper() for i in Gene_map]
    Prot_map = mapping[mapping.columns[1]].tolist()

    Gene_2_num = {}
    for m, n in enumerate(GeneID_t):
        if n in Gene_map:
            Gene_2_num[n] = m

    Prot_2_num = {}
    for m, n in enumerate(Prot_ID):
        
        if n in Prot_map:
            Prot_2_num[n] = m
            
    map_2_num = {Gene_2_num[i]:Prot_2_num[j] for i, j in zip(Gene_map, Prot_map) if (i in Gene_2_num.keys() and j in Prot_2_num.keys())}
    print(f'the num of map: {len(Prot_map)} ---the num of geneID in mapping: {len(Gene_2_num)} and the num of map pairs: {len(map_2_num)}')


    # mask the Gene data to get more samples, and find out the corresponding
    data = []
    target = []
    target_classify = []
    info = []
    n = size   # how many samples for each data point

    for dat, tar in zip(Gene_all, Prot_all):
        Gene_dat = dat  # gene data
        tar = torch.tensor(tar, dtype=torch.float32)  # protein data

        P_lar = tar[tar > 0.8]
        tar_clss = torch.as_tensor([1 if x>0.8 else 0 for x in tar])
        print(f'------- The number of original proterin: {tar.shape} and high expressed protein: {P_lar.shape} --------\n')
        print(tar_clss.shape)
        # print(tar)
        if pertage == 0.0:
            gene_mask = list(map_2_num.keys())
            prot_mask = [map_2_num[i] for i in gene_mask]
            genes_2_proten = torch.tensor(np.array([gene_mask, prot_mask]), dtype=torch.int64)

            Gene_dat = torch.tensor(Gene_dat, dtype=torch.float32)


            data.append(Gene_dat)
            target.append(tar)
            target_classify.append(tar_clss)
            info.append(genes_2_proten)


        elif stage == None:
            for i in range(n):
                # random mask any gene IDs, and find out the gene indeces in the mapping pool
                gene_mask = np.random.choice(list(map_2_num.keys()), replace=False, size=int(len(Gene_2_num) * pertage))
                Gene_dat[gene_mask] = 0.0
                Gene_dat = torch.tensor(Gene_dat, dtype=torch.float32)
                gene_mask = list(map_2_num.keys())
                prot_mask = [map_2_num[i] for i in gene_mask]
                genes_2_proten = torch.tensor(np.array([gene_mask, prot_mask]), dtype=torch.int64)

                data.append(Gene_dat)
                target.append(tar)
                target_classify.append(tar_clss)
                info.append(genes_2_proten)

        else:
            for i in range(n):
                # random mask any gene IDs, and find out the gene indeces in the mapping pool
                gene_mask = np.random.choice(list(map_2_num.keys()), replace=False, size=int(len(Gene_2_num) * pertage))
                Gene_dat[gene_mask] = 0.0
                Gene_dat = torch.tensor(Gene_dat, dtype=torch.float32)

                # find ou the corresponding protein indeces
                prot_mask = [map_2_num[i] for i in gene_mask]
                genes_2_proten = torch.tensor(np.array([gene_mask, prot_mask]), dtype=torch.int64)

                info.append(genes_2_proten)
                data.append(Gene_dat)
                target.append(tar)
                target_classify.append(tar_clss)
                '''
                # find out the gene indeces in the mapping pool
                gene_ind = [i for i in gene_mask if i in map_2_num.keys]
                # find ou the corresponding protein indeces
                Prot_ind = [map_2_num[i] for i in gene_ind]
                '''


    data = torch.stack(data, 0)
    target = torch.stack(target, 0)
    target_classify = torch.stack(target_classify,0)
    # plotTime(target)
    info = torch.stack(info, 0)
    print(f'the total number of samples we have: {data.shape} and protein shape: {target.shape} ,{target_classify.shape} ----')
    print(info.shape)

    '''
    input = data.numpy()
    output = target.numpy()
    index = info.numpy()
    
    # save data to retrain on same datasets
    name_in = dir_in + stage + '_' + str(pertage)+'_'+str(size)
    name_out = dir_out + stage + '_' + str(pertage)+'_'+str(size)
    name_index = dir_index + stage + '_' + str(pertage)+'_'+str(size)
    np.save(root+name_in, input)
    np.save(root+name_out, output)
    np.save(root+name_index, index)
    '''

    return data, target, info, target_classify

class Data2target_test(Dataset):
    def __init__(self, stage = 'train', size = 10, pertage = 0.5):
        if os.path.exists(root+dir_index+ stage +'_'+ str(pertage)+'_'+str(size)+'.npy'):
            print("========== load dataset ================")

            self.data = np.load(root+dir_in + stage +'_'+ str(pertage)+'_'+str(size)+'.npy')
            self.target = np.load(root+dir_out + stage +'_'+ str(pertage)+'_'+str(size)+'.npy')
            self.info = np.load(root+dir_index + stage +'_'+ str(pertage)+'_'+str(size)+'.npy')
            self.target_classify = np.load(root+dir_out + stage +'_'+ str(pertage)+'_'+str(size)+'.npy')

            self.data = torch.from_numpy(self.data)
            self.target = torch.from_numpy(self.target)
            self.info = torch.from_numpy(self.info)
            self.target_classify = torch.from_numpy(self.target_classify)

        else:
            self.data, self.target, self.info, self.target_classify = gene2protein(stage = stage, size = size, pertage = pertage)

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx], self.info[idx],self.target_classify[idx]