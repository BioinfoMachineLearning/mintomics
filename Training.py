import os
import glob
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import numpy as np
from pandas import read_csv
from torch import  Tensor
import torch.nn.functional as F
import re
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import wandb
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryConfusionMatrix, BinaryF1Score, MultilabelAccuracy, MultilabelF1Score, MultilabelConfusionMatrix,MultilabelPrecision,MultilabelRecall

from torchmetrics.regression import MeanSquaredError,R2Score,MeanAbsoluteError

from Model import TransformerMintomics
from argparse import ArgumentParser
import scipy.signal as signal
from src.dataset.PrepareDataset import Psedu_data, Data2target, gene2protein
from src.dataset.prepare_test_data import Data2target_test


AVAIL_GPUS = [1,2]
NUM_NODES = 1
BATCH_SIZE = 1
DATALOADERS = 1
ACCELERATOR = "gpu"
EPOCHS = 1
ATT_HEAD = 1
ENCODE_LAYERS = 1
DATASET_DIR = "./"

#label_dict = read_csv(DATASET_DIR+"/Dataset/Labels_proc/Labels_control.csv",index_col=0)
#label_dict = read_csv(DATASET_DIR+"/Dataset/Labels_proc/Labels_control.csv",index_col=0)
#Num_classes = len(label_dict)
Num_classes = 3060
"""

torch.set_default_tensor_type(torch.FloatTensor)  # Ensure that the default tensor type is FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose the device you want to use

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner to find the best algorithm to use for hardware
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # Set the default tensor type to CUDA FloatTensor
    torch.set_float32_matmul_precision('medium')  # Set Tensor Core precision to medium

"""

CHECKPOINT_PATH = f"{DATASET_DIR}/Trainings/tempo"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


class Mintomics(pl.LightningModule):
    def __init__(self, learning_rate=1e-4,attn_head=ATT_HEAD,encoder_layers=ENCODE_LAYERS,n_class=1, **model_kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        self.model = TransformerMintomics(attn_head=attn_head,encoder_layers=encoder_layers,n_class=n_class,**model_kwargs)
        self.loss_fn = nn.BCEWithLogitsLoss()
       
        self.metrics_class = MetricCollection([MultilabelAccuracy(num_labels=76,average='micro'),
                                              MultilabelPrecision(num_labels=76,average='micro'),
                                              MultilabelF1Score(num_labels=76,average='micro'),
                                              MultilabelRecall(num_labels=76,average='micro')])
        #self.metrics_class = MetricCollection([BinaryAccuracy(),
        #                                 BinaryPrecision(),
        #                                 BinaryRecall(),
        #                                 BinaryF1Score()])

        self.train_metrics_class = self.metrics_class.clone(prefix="train_")
  
        self.valid_metrics_class = self.metrics_class.clone(prefix="valid_")
  
        self.test_metrics_class = self.metrics_class.clone(prefix="test_")
  

    def forward(self, pest_sample):
        x = self.model(pest_sample)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=20, eps=1e-10, verbose=True)
        metric_to_track = 'valid_loss'
        return{'optimizer':optimizer,
               'lr_scheduler':lr_scheduler,
               'monitor':metric_to_track}
    
    def training_step(self,batch,batch_idx):
        batch_data = batch[0]
        
        
        inf = batch[2]
        #print(batch_data.shape)
        y_hat,_ = self.forward(batch_data)
        ##print(y_hat.shape)
        batch_label_class = batch[3].cuda()
        
        class_pred = y_hat[:, inf[0, 1]]
        
        #target = torch.transpose(batch_label_class, 1, 2)
        target = batch_label_class[:, inf[0, 1]]
       
        #batch_label_class = batch_label_class[:,None].cuda()

        #class_pred = y_hat.view(-1) 
        
        loss_class = self.loss_fn(class_pred,target.float())
        metric_log_class = self.train_metrics_class(class_pred, target)
        self.log_dict(metric_log_class)
        loss = (loss_class)
        self.log('train_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        batch_data = batch[0]
        
        
        inf = batch[2]
        #print(batch_data.shape)
        y_hat,_ = self.forward(batch_data)
        ##print(y_hat.shape)
        batch_label_class = batch[3].cuda()
        
        class_pred = y_hat[:, inf[0, 1]]
        
        #target = torch.transpose(batch_label_class, 1, 2)
        target = batch_label_class[:, inf[0, 1]]
       
        #batch_label_class = batch_label_class[:,None].cuda()

        #class_pred = y_hat.view(-1) 
        
        loss_class = self.loss_fn(class_pred,target.float())
        metric_log_class = self.valid_metrics_class(class_pred, target)
        self.log_dict(metric_log_class)
        loss = (loss_class)
        self.log('valid_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss
       
    
    def test_step(self,batch, batch_idx):
        batch_data = batch[0]
        inf = batch[2]
        y_hat,attnt = self.forward(batch_data)
        batch_label_class = batch[3].cuda()
        
        class_pred = y_hat[:, inf[0, 1]]
        
        #target = torch.transpose(batch_label_class, 1, 2)
        target = batch_label_class[:, inf[0, 1]]
       
        #batch_label_class = batch_label_class[:,None].cuda()

        #class_pred = y_hat.view(-1) 
        
        loss_class = self.loss_fn(class_pred,target.float())
        metric_log_class = self.test_metrics_class(class_pred, target)
        self.log_dict(metric_log_class)
        loss = (loss_class)
        self.log('test_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
       
        #conf_mat = BinaryConfusionMatrix().to("cuda")
        #conf_vals = conf_mat(class_pred, batch_label_class.squeeze())
        #print("Test Data Confusion Matrix: \n")
        #print(conf_vals)
        return {f'preds_class' : class_pred, f'targets_class' :target,f'attention':attnt,f'inf':inf}
        
    def test_epoch_end(self, outputs):
        # Log individual results for each dataset
        
        #for i  in range(len(outputs)):
            dataset_outputs = outputs
            
            #torch.save(dataset_outputs,"Predictions.pt")
            class_preds = torch.cat([x[f'preds_class'] for x in dataset_outputs])
            class_targets = torch.cat([x[f'targets_class'] for x in dataset_outputs])
            conf_mat = BinaryConfusionMatrix().cuda()
            conf_vals = conf_mat(class_preds, class_targets)
            fig = sns.heatmap(conf_vals.cpu() , annot=True, cmap="Blues", fmt="d")
            ind = torch.nonzero(class_targets[0,:]>0.5)
            attention = torch.cat([x[f'attention'] for x in dataset_outputs]).squeeze()
            inf = torch.cat([x[f'inf'] for x in dataset_outputs]).squeeze()
            attention1 = attention[:,inf[0,:]]
            attention2 = attention1[:,ind].squeeze()
            print(inf.shape, attention2.shape)
            # Get top 100 genes along rows for all columns
            top_genes_values, top_genes_indices = torch.topk(attention2, k=20, dim=0)
            mask = torch.zeros_like(attention2)
            
            mask[top_genes_indices, torch.arange(attention2.shape[1])] = 1.0
            print(mask)
            # Multiply the mask with the selected portion to keep only the top genes values
            attention2 = attention2 * mask
            # Calculate the hierarchical clustering
            # Calculate the hierarchical clustering
            #row_linkage = hierarchy.linkage(attention2, method='average')
            #col_linkage = hierarchy.linkage(attention2.T, method='average')

            # Reorder the matrix rows and columns based on the clustering
            #idx_row = hierarchy.dendrogram(row_linkage, no_plot=True)['leaves']
            #idx_col = hierarchy.dendrogram(col_linkage, no_plot=True)['leaves']
            # Calculate the hierarchical clustering
            #row_clusters = fastcluster.linkage(attention2, method='average')
            #col_clusters = fastcluster.linkage(attention2.T, method='average')
            
            # Plot the dendrogram for rows
            fig1 = plt.figure(figsize=(10, 20))
            sns.heatmap(attention2, cmap='rocket_r')
            plt.show()
            # Plot the reordered matrix
            #fig1 = plt.figure(figsize=(10, 10))
            #sns.clustermap(attention2, cmap='bone')
            #plt.show()
            #attention = self.model.encod.self_attn.
            #fig1 = plt.figure(figsize=(50, 100))
            #ax = fig1.add_subplot(111)
            #cax = ax.matshow(attention2.cpu().numpy(), cmap='bone')
            #cax.autoscale()
            #fig1.colorbar(cax)
            
            wandb.log({f"conf_mat" : wandb.Image(fig),"attentions":wandb.Image(fig1)})
            
            return super().test_epoch_end(outputs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--attn_head',type=int,default=ATT_HEAD)
        parser.add_argument('--encoder_layers',type=int,default=ENCODE_LAYERS)
        parser.add_argument('--n_class',type=int,default=1)
        return parser


def train_mintomics_classifier():
    pl.seed_everything(42)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Mintomics.add_model_specific_args(parser)
    parser.add_argument('--num_gpus', type=int, default=AVAIL_GPUS,
                        help="Number of GPUs to use (e.g. -1 = all available GPUs)")
    parser.add_argument('--nodes', type=int, default=NUM_NODES, help="Number of nodes to use")
    parser.add_argument('--num_epochs', type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int,
                        help="effective_batch_size = batch_size * num_gpus * num_nodes")
    parser.add_argument('--num_dataloader_workers', type=int, default=DATALOADERS)
    parser.add_argument('--entity_name', type=str, default='aghktb', help="Weights and Biases entity name")
    parser.add_argument('--project_name', type=str, default='Mintomics',
                        help="Weights and Biases project name")
    parser.add_argument('--save_dir', type=str, default=CHECKPOINT_PATH, help="Directory in which to save models")

    parser.add_argument('--unit_test', type=int, default=False,
                        help="helps in debug, this touches all the parts of code."
                             "Enter True or num of batch you want to send, " "eg. 1 or 7")
    args = parser.parse_args()
    
    args.devices = args.num_gpus
    args.num_nodes = args.nodes
    args.accelerator = ACCELERATOR
    args.max_epochs = args.num_epochs
    args.fast_dev_run = args.unit_test
    args.log_every_n_steps = 1
    args.detect_anomaly = True
    args.enable_model_summary = True
    args.weights_summary = "full"
    save_PATH = DATASET_DIR+"/Trainings/"+args.save_dir
    os.makedirs(save_PATH, exist_ok=True)

    # load data and get corresponding information
    dataset_train = Data2target(stage='train', size = 4000, pertage = 0.15)  #100 samples each dp
    dataset_valid = Data2target(stage='valid', size = 1000, pertage = 0.15) 
    dataset_test = Data2target_test(stage='test',size=2000,pertage=0.15)
    #train_size = int(0.7 * len(dataset))
    #val_size = int(0.1 * len(dataset))
    #test_size = len(dataset) - (train_size+val_size)
    #dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])
    
    #dataset_valid = MicrographDataValid(DATASET_DIR)
    #dataset_test = MicrographDataValid(DATASET_DIR) # using validation data for testing here
    train_loader = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_dataloader_workers)
    
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_dataloader_workers)
    test_loader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_dataloader_workers)
   # torch.save(test_loader,DATASET_DIR+'/test.pt')
    model = Mintomics(learning_rate=1e-4,n_class=Num_classes)
    
    trainer = pl.Trainer.from_argparse_args(args)
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss', save_top_k=10, dirpath=save_PATH, filename='mintomics_{epoch:02d}_{valid_loss:6f}')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping_callback = EarlyStopping(monitor='valid_loss', mode='min', min_delta=0.0, patience=30)
    trainer.callbacks = [checkpoint_callback, lr_monitor, early_stopping_callback]
    logger = WandbLogger(project=args.project_name, entity=args.entity_name,name=args.save_dir, offline=False, save_dir=".")
    trainer.logger = logger
    wandb.init()
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(dataloaders=test_loader, ckpt_path='best')
   



if __name__ == "__main__":
    train_mintomics_classifier()
    wandb.finish()
