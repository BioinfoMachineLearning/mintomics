import torch.nn as nn
import torch
from torch import  Tensor
import math
class TransformerMintomics(nn.Module):
    def __init__(self, dim_model=7, attn_head=1, dim_ff=64, drop=0.1, batch_f=True, encoder_layers=1,n_class=1):
        super(TransformerMintomics,self).__init__()
        self.encod = nn.TransformerEncoderLayer(d_model=dim_model, nhead=attn_head, dim_feedforward=dim_ff, dropout=drop, 
                                                                    batch_first=batch_f)
        self.layernorm = nn.LayerNorm(dim_model)
       
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=attn_head, dim_feedforward=dim_ff, dropout=drop, 
                                                                    batch_first=batch_f)
                                                                  
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer,num_layers=encoder_layers)
        #self.MLPreg = nn.Sequential(nn.Linear(172*dim_model*2,dim_model*4),nn.ReLU(),nn.BatchNorm1d(dim_model*4),
        #                           nn.Linear(dim_model*4,dim_model),nn.ReLU(),nn.BatchNorm1d(dim_model),nn.Linear(dim_model,1),nn.ReLU())
        self.MLPclass = nn.Sequential(nn.Linear(dim_model,dim_model*2),nn.ReLU(),nn.BatchNorm1d(dim_model*2),
                                      nn.Linear(dim_model*2, dim_model*8),nn.ReLU(),nn.BatchNorm1d(dim_model*8),nn.Linear(dim_model*8,n_class),nn.functional.sigmoid())
        self.activation_ELU= nn.ReLU()

        

    def forward(self,x):
        x = x.cuda()

        x = x.view(x.size(0), x.size(2), x.size(1))
        x = self.encod(x)
        x = self.layernorm(x)
        x = self.activation_ELU(x)
        x = self.transformer_encoder(x)
        x,_ = torch.max(x, dim=1)
        x = x.squeeze(0)     
        x = self.activation_ELU(x)
 
 #       x = x.view(x.size(0), x.size(2)*x.size(1))
       
        classify = self.MLPclass(x)

        
        return classify