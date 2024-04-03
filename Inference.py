import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import os
import glob
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import numpy as np
from pandas import read_csv,DataFrame,concat
from torch import  Tensor
import torch.nn.functional as F
import re
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
import wandb
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryConfusionMatrix, BinaryF1Score,MulticlassAccuracy, MultilabelAccuracy, MultilabelF1Score, MultilabelConfusionMatrix,MultilabelPrecision,MultilabelRecall,MulticlassPrecision,MulticlassRecall,MulticlassF1Score,MulticlassConfusionMatrix
from torchmetrics.regression import MeanSquaredError,R2Score,MeanAbsoluteError

from Model import TransformerMintomics
from argparse import ArgumentParser
import scipy.signal as signal
from PrepareDataset import Psedu_data , Data2target,gene2protein
from prepare_test_data import Data2target_test
from scipy.cluster import hierarchy
from Diff_Gene_proc import selected_genes,significant_proteins,common_genes
import itertools
DATA_DIR = '/home/aghktb/JOYS_Project/mintomics'
root = '/bmlfast/joy_RNA/Data/'
dir_in = 'bulkRNA_p_'
dir_out = 'protein_p_'
dir_index = 'mapping_p_'
AVAIL_GPUS = [1,2]
NUM_NODES = 1
BATCH_SIZE = 1
DATALOADERS = 1
ACCELERATOR = "gpu"
EPOCHS = 3
ATT_HEAD = 1
ENCODE_LAYERS = 2
DATASET_DIR = "/home/aghktb/JOYS_PROJECT/mintomics"

#label_dict = read_csv(DATASET_DIR+"/Dataset/Labels_proc/Labels_control.csv",index_col=0)
#label_dict = read_csv(DATASET_DIR+"/Dataset/Labels_proc/Labels_control.csv",index_col=0)
#Num_classes = len(label_dict)
Num_classes = 3060

CHECKPOINT_PATH = f"{DATASET_DIR}/Trainings/tempo"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
class Mintomics(pl.LightningModule):
    def __init__(self, learning_rate=1e-4,attn_head=ATT_HEAD,encoder_layers=ENCODE_LAYERS,n_class=1, **model_kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        self.model = TransformerMintomics(attn_head=attn_head,encoder_layers=encoder_layers,n_class=n_class,**model_kwargs)
        self.loss_fn = nn.BCEWithLogitsLoss()
       
        self.metrics_class1 = MetricCollection([BinaryAccuracy(),
                                         BinaryPrecision(),
                                         BinaryRecall(),
                                         BinaryF1Score()])

        self.metrics_class = MetricCollection([MultilabelAccuracy(num_labels=2406,average='micro'),
                                              MultilabelPrecision(num_labels=2406,average='micro'),
                                              MultilabelF1Score(num_labels=2406,average='micro'),
                                              MultilabelRecall(num_labels=2406,average='micro')])
        self.test_metrics_class = self.metrics_class.clone(prefix="test_")
        self.test_metrics_class1 = self.metrics_class1.clone(prefix="test_")
    def forward(self, pest_sample):
        x = self.model(pest_sample)
        return x
    
    def test_step(self,batch, batch_idx):
        batch_data = batch[0]
        inf = batch[2]
        tfs = batch_data[:,:,6]
        
        y_hat,attnt = self.forward(batch_data)
        batch_label_class = batch[3]
        
        class_pred = y_hat[:, inf[0, 1]]
        
        #target = torch.transpose(batch_label_class, 1, 2)
        target = batch_label_class[:, inf[0, 1]]
        
        
        
        #target = target_1[:,ind]
        #class_pred = class_pred_1[:,ind]
        print(class_pred.shape,target.shape)
        #batch_label_class = batch_label_class[:,None].cuda()

        #class_pred = y_hat.view(-1) 
        
        loss_class = self.loss_fn(class_pred,target.float())
        #metric_log_class = self.test_metrics_class(class_pred, target)
        #self.log_dict(metric_log_class)
        metric_log_class1 = self.test_metrics_class1(class_pred, target)
        self.log_dict(metric_log_class1)
        loss = (loss_class)
        self.log('test_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
       
        #conf_mat = BinaryConfusionMatrix().to("cuda")
        #conf_vals = conf_mat(class_pred, batch_label_class.squeeze())
        #print("Test Data Confusion Matrix: \n")
        #print(conf_vals)
        return {f'preds_class' : class_pred, f'targets_class' :target,f'attention':attnt,f'inf':inf,f'tfs':tfs}
    def construct_networkx(edge_weight_matrix):
            """
              Constructs a network from an edge weight matrix.

              Args:
                edge_weight_matrix: A square matrix of edge weights.

              Returns:
                A `torch_geometric.data.Data` object representing the network.
            """

            edge_index = (abs(edge_weight_matrix) > 0.5).nonzero().t()
            row, col = edge_index
            edge_weight = edge_weight_matrix[row, col]
            #G = nx.Graph(np.matrix(edge_weight_matrix))


            # Create a Data object to represent the graph.
            data = Data(edge_index=edge_index, edge_weight=edge_weight)


              # Return the data object.
            return data  
    def test_epoch_end(self, outputs):
        # Log individual results for each dataset
        
        #for i  in range(len(outputs)):
            dataset_outputs = outputs
            
            #torch.save(dataset_outputs,"Predictions.pt")
            gene_names = read_csv("/home/aghktb/JOYS_Project/mintomics/Dataset/Data_cpm/Data_2_5preg.csv",)
            
            #print(gene_names)
            gene_name = gene_names["Unnamed: 0"]
            gene_name1 = gene_names["Unnamed: 0"].tolist()
            print(gene_name.head())
            #get differentially significant genes
            sele_genes = list(set(selected_genes))
            
            diff_gene_ind = [gene_name1.index(gene) for gene in sele_genes]
            #print(diff_gene_ind)

            
            class_preds = torch.cat([x[f'preds_class'] for x in dataset_outputs])
            class_targets = torch.cat([x[f'targets_class'] for x in dataset_outputs])
            conf_mat = BinaryConfusionMatrix()
            conf_vals = conf_mat(class_preds, class_targets)
            fig = sns.heatmap(conf_vals.cpu() , annot=True, cmap="Blues", fmt="d")
            ind = torch.nonzero(class_targets[0,:]>0.8)
            attention = torch.cat([x[f'attention'] for x in dataset_outputs]).squeeze()
            inf = torch.cat([x[f'inf'] for x in dataset_outputs]).squeeze()

            Tfs =  torch.cat([x[f'tfs'] for x in dataset_outputs]).squeeze()
            
            tf_ind = np.nonzero(Tfs>0).numpy()
            
            #tf_names = gene_name.values[tf_ind].tolist()
            diff_tf_ind = torch.tensor(tf_ind[np.isin(tf_ind, diff_gene_ind)]).T.numpy()
            print((tf_ind),(diff_tf_ind))
            #print(diff_tf_ind)
            tf_names = list(itertools.chain.from_iterable(gene_name.values[tf_ind].tolist()))
            diff_tf_names = list(itertools.chain.from_iterable(gene_name.values[diff_tf_ind].tolist()))
            #print((tf_names))
            print((tf_names),(diff_tf_names))
            protein_gene_names = gene_name.values[inf[0]]
            ###differential significan protein index
            protein_diff_gene_name = [idx for idx in protein_gene_names if idx in common_genes]
            protein_diff_gene_ind = [gene_name1.index(gene) for gene in common_genes]
            print(len(protein_diff_gene_name))
            print(len(ind),len(protein_diff_gene_ind))
            diff_ind = [ind[np.isin(ind, protein_diff_gene_ind)]]
            print(len(diff_ind))
            high_proteins = list(itertools.chain.from_iterable(protein_gene_names[ind].tolist()))
            #print(len(high_proteins))
            #print(inf.shape)
            
            attention1 = attention.fill_diagonal_(0)
            

            attention1 = attention1[:,inf[0]]
            attention2 = attention1[:,ind].squeeze()
            attention3 = attention2[diff_tf_ind,:].squeeze()
            print(attention3.shape)
            atten_sig = torch.sigmoid(attention3)
            df = DataFrame(atten_sig)
            df.index = diff_tf_names
            df.columns = high_proteins
            print(tf_names)
            #df.to_csv("/home/aghktb/JOYS_PROJECT/mintomics/Tfs_highprot_control_adj.csv")
            #print(inf.shape, attention2.shape)
            # Get top 100 genes along rows for all columns
            #top_genes_values, top_genes_indices = torch.topk(attention3, k=500, dim=0)
            #print(top_genes_indices.shape)
            #mask = torch.zeros_like(attention3)
            #print(top_genes_values)
            #mask[top_genes_indices, torch.arange(attention3.shape[1])] = 1.0*10000
            
            #print(mask)
            # Multiply the mask with the selected portion to keep only the top genes values
            #attention3 = attention3 * mask
            
            top_tf_values, top_tf_indices = torch.topk(attention3, k=10, dim=0)
            #print(top_tf_indices.shape)
            attention4 = attention3[top_tf_indices,torch.arange(attention3.shape[1])]*10000
            atten_sig = torch.sigmoid(attention4)


            
            top_tfs_names = [[diff_tf_names[idx] for idx in gene_top_indices] for gene_top_indices in top_tf_indices.T]
            print(len(top_tfs_names))
            tf_df = DataFrame(top_tfs_names)
            tf_df.index = high_proteins
            print(tf_df.T)
            tf_df_T = tf_df.T
            top_values_df = DataFrame(top_tf_values*10000)
            top_values_df.columns = high_proteins
            print(top_values_df)
            tf_names_att_df = concat([tf_df_T, top_values_df], axis=1, join='outer')
            print(tf_names_att_df)
            #tf_names_att_df.to_csv("/home/aghktb/JOYS_PROJECT/mintomics/Tfs_highprot_control_n.csv",index=None)
            
            
           
            # Plot the dendrogram for rows
            
            fig1 = plt.figure(figsize=(40, 40))
            sns.heatmap(atten_sig, cmap='rocket_r')
            plt.title("Scaled Attentions of Top 500  Genes Influencing Protein-Coding Gene Expressions")
            plt.xlabel("All Genes ")
            plt.ylabel("Protein-Coding Genes")
            plt.show()
            
            wandb.log({f"conf_mat" : wandb.Image(fig),"attentions":wandb.Image(fig1)})
            
            return super().test_epoch_end(outputs)
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--attn_head',type=int,default=ATT_HEAD)
        parser.add_argument('--encoder_layers',type=int,default=ENCODE_LAYERS)
        parser.add_argument('--n_class',type=int,default=1)
        parser.add_argument('--save_dir', type=str, default=CHECKPOINT_PATH, help="Directory in which to save models")
        parser.add_argument('--chkpt',type=str,help="Checkpoint name")
        return parser
def train_mintomics_classifier():
    pl.seed_everything(42)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = ArgumentParser()
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Mintomics.add_model_specific_args(parser)
    parser.add_argument('--entity_name', type=str, default='aghktb', help="Weights and Biases entity name")
    parser.add_argument('--project_name', type=str, default='Mintomics',
                        help="Weights and Biases project name")
    args = parser.parse_args()
    check_pt_dir = args.save_dir
    dataset_test = Data2target_test(stage='test',size=1,pertage=0)
    test_loader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=DATALOADERS)
    model = Mintomics(learning_rate=1e-4,n_class=Num_classes)
    trainer = pl.Trainer.from_argparse_args(args)
    logger = WandbLogger(project=args.project_name, entity=args.entity_name,name=args.save_dir+"_test_differential", offline=False, save_dir=".")
    trainer.logger = logger
    pest_checkpoint = DATASET_DIR+"/Trainings/"+args.save_dir+"/"+args.chkpt
    trainer.test(model, dataloaders=test_loader, ckpt_path=pest_checkpoint)

if __name__ == "__main__":
    train_mintomics_classifier()
    
