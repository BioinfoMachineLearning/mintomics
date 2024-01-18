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
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryConfusionMatrix, BinaryF1Score,MulticlassAccuracy, MultilabelAccuracy, MultilabelF1Score, MultilabelConfusionMatrix,MultilabelPrecision,MultilabelRecall,MulticlassPrecision,MulticlassRecall,MulticlassF1Score,MulticlassConfusionMatrix
from torchmetrics.regression import MeanSquaredError,R2Score,MeanAbsoluteError

from Model import TransformerMintomics
from argparse import ArgumentParser
import scipy.signal as signal
from PrepareDataset import Psedu_data , Data2target,gene2protein
from prepare_test_data import Data2target_test
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
        y_hat,attnt = self.forward(batch_data)
        batch_label_class = batch[3]
        
        class_pred = y_hat[:, inf[0, 1]]
        
        #target = torch.transpose(batch_label_class, 1, 2)
        target = batch_label_class[:, inf[0, 1]]
        
        #batch_label_class = batch_label_class[:,None].cuda()

        #class_pred = y_hat.view(-1) 
        
        loss_class = self.loss_fn(class_pred,target.float())
        metric_log_class = self.test_metrics_class(class_pred, target)
        self.log_dict(metric_log_class)
        metric_log_class1 = self.test_metrics_class1(class_pred, target)
        self.log_dict(metric_log_class1)
        loss = (loss_class)
        self.log('test_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
       
        #conf_mat = BinaryConfusionMatrix().to("cuda")
        #conf_vals = conf_mat(class_pred, batch_label_class.squeeze())
        #print("Test Data Confusion Matrix: \n")
        #print(conf_vals)
        return {f'preds_class' : class_pred, f'targets_class' :target,f'attention':attnt}
        
    def test_epoch_end(self, outputs):
        # Log individual results for each dataset
        
        #for i  in range(len(outputs)):
            dataset_outputs = outputs
            
            #torch.save(dataset_outputs,"Predictions.pt")
            class_preds = torch.cat([x[f'preds_class'] for x in dataset_outputs])
            class_targets = torch.cat([x[f'targets_class'] for x in dataset_outputs])
            conf_mat = BinaryConfusionMatrix()
            conf_vals = conf_mat(class_preds, class_targets)
            fig = sns.heatmap(conf_vals.cpu() , annot=True, cmap="Blues", fmt="d")
            attention = torch.cat([x[f'attention'] for x in dataset_outputs]).squeeze()
            
            #attention = self.model.encod.self_attn.
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            cax = ax.matshow(attention.cpu().numpy(), cmap='bone')
            fig1.colorbar(cax)
            
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
    logger = WandbLogger(project=args.project_name, entity=args.entity_name,name=args.save_dir+"_test2", offline=False, save_dir=".")
    trainer.logger = logger
    pest_checkpoint = DATASET_DIR+"/Trainings/"+args.save_dir+"/"+args.chkpt
    trainer.test(model, dataloaders=test_loader, ckpt_path=pest_checkpoint)

if __name__ == "__main__":
    train_mintomics_classifier()
