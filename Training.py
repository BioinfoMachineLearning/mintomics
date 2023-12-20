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
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryConfusionMatrix, BinaryF1Score,MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,MulticlassF1Score,MulticlassConfusionMatrix
from torchmetrics.regression import MeanSquaredError,R2Score,MeanAbsoluteError

from Model import TransformerMintomics
from argparse import ArgumentParser
import scipy.signal as signal
from Datamaker import Psedu_data 
AVAIL_GPUS = [0]
NUM_NODES = 1
BATCH_SIZE = 2
DATALOADERS = 1
ACCELERATOR = "gpu"
EPOCHS = 2
ATT_HEAD = 1
ENCODE_LAYERS = 2
DATASET_DIR = "/home/aghktb/JOYS_PROJECT/mintomics"

#label_dict = read_csv(DATASET_DIR+"/Dataset/Labels_proc/Labels_control.csv",index_col=0)
#label_dict = read_csv(DATASET_DIR+"/Dataset/Labels_proc/Labels_control.csv",index_col=0)
#Num_classes = len(label_dict)
Num_classes = 10
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
        self.loss_fn = nn.BCELoss()
       
        self.metrics_class = MetricCollection([BinaryAccuracy(),
                                         BinaryPrecision(),
                                         BinaryRecall(),
                                         BinaryF1Score()])

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
        print(batch_data.shape)
        y_hat = self.forward(batch_data)
        
        batch_label_class = batch[1].cuda()
        
        class_pred = y_hat[:, inf[0, 0]].cuda()
   
        target = torch.transpose(batch_label_class, 1, 2)
        target = target[:, inf[0, 1],:].squeeze()
        print(class_pred.shape)
        print(target.shape)
        #batch_label_class = batch_label_class[:,None].cuda()

        #class_pred = y_hat.view(-1) 
        
        loss_class = self.loss_fn(class_pred,target)
        metric_log_class = self.train_metrics_class(class_pred, target)
        self.log_dict(metric_log_class)
        loss = (loss_class)
        self.log('train_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        batch_data = batch[0]
        
        
        inf = batch[2]
        print(batch_data.shape)
        y_hat = self.forward(batch_data)
        print(y_hat.shape)
        batch_label_class = batch[1].type('torch.LongTensor').T.cuda()
        #class_pred = y_hat[:, inf[0, 0], :]
   
        #target = torch.transpose(batch_label_class, 1, 2)
        #target = target[:, inf[0, 1], :]
       
        #batch_label_class = batch_label_class[:,None].cuda()

        class_pred = y_hat.view(-1) 
        print(class_pred)
        loss_class = self.loss_fn(class_pred,batch_label_class.view(-1).float())
        metric_log_class = self.valid_metrics_class(class_pred, batch_label_class.int().squeeze())
        self.log_dict(metric_log_class)
        loss = (loss_class)
        self.log('train_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss
       
    
    def test_step(self,batch, batch_idx):
        batch_data = batch[0]
        inf = batch[2]
        y_hat = self.forward(batch_data)
        batch_label_class = batch[1].type('torch.LongTensor')
        
        batch_label_class = batch_label_class[:,None].cuda()

        class_pred = y_hat[:, inf[0, 0], :]
   
        target = torch.transpose(batch_label_class, 1, 2)
        target = target[:, inf[0, 1], :]
       
        loss_class = self.loss_fn(class_pred,target)
        metric_log_class = self.test_metrics_class(class_pred, batch_label_class.int().squeeze())
        self.log_dict(metric_log_class)
        loss = (loss_class)
        self.log('train_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
       
        conf_mat = BinaryConfusionMatrix(num_classes=self.hparams.n_class).to("cuda")
        conf_vals = conf_mat(class_pred, batch_label_class.squeeze())
        print("Test Data Confusion Matrix: \n")
        print(conf_vals)
        

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

    dataset = Psedu_data()
    print(len(dataset))
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - (train_size+val_size)
    dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])
    
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
    #logger = WandbLogger(project=args.project_name, entity=args.entity_name,name=args.save_dir, offline=False, save_dir=".")
    #trainer.logger = logger
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(dataloaders=test_loader, ckpt_path='best')
   



if __name__ == "__main__":
    train_mintomics_classifier()