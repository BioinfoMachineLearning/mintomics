import torch.nn as nn
import torch
from torch import  Tensor
import math
class TransformerMintomics(nn.Module):
    def __init__(self, dim_model=7, attn_head=1, dim_ff=32, drop=0.5, batch_f=True, encoder_layers=1,n_class=1):
        super(TransformerMintomics,self).__init__()
        self.encod = nn.TransformerEncoderLayer(d_model=dim_model, nhead=attn_head, dim_feedforward=dim_ff, dropout=drop, 
                                                                    batch_first=batch_f)
        self.layernorm = nn.LayerNorm(dim_model)
       
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=attn_head, dim_feedforward=dim_ff, dropout=drop, 
                                                                    batch_first=batch_f)
                                                                  
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer,num_layers=encoder_layers)
        #self.MLPreg = nn.Sequential(nn.Linear(172*dim_model*2,dim_model*4),nn.ReLU(),nn.BatchNorm1d(dim_model*4),
        #                           nn.Linear(dim_model*4,dim_model),nn.ReLU(),nn.BatchNorm1d(dim_model),nn.Linear(dim_model,1),nn.ReLU())
        self.MLPclass = nn.Sequential(nn.Linear(dim_model,dim_model*128),nn.ReLU(),#nn.BatchNorm1d(dim_model*2),
                                      nn.Linear(dim_model*128, dim_model*32),nn.ReLU(),nn.Linear(dim_model*32,n_class))#nn.BatchNorm1d(dim_model*8)
        self.activation_ELU= nn.ReLU()

        

    def forward(self,x):
        x = x.cuda()

        #x = x.view(x.size(0), x.size(2), x.size(1))
        
        x = self.encod(x)
        #print(x.shape)
        x = self.layernorm(x)
        x = self.activation_ELU(x)
        x = self.transformer_encoder(x)
        #print(x.shape)
        x = torch.mean(x, dim=1)
        
           
        x = self.activation_ELU(x)
 
 #       x = x.view(x.size(0), x.size(2)*x.size(1))
       
        classify = nn.functional.sigmoid(self.MLPclass(x))
        #classify = self.MLPclass(x)
        
        
        return classify