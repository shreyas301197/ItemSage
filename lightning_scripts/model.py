import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from random import sample
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import random
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torchmetrics

class ItemSageModel(pl.LightningModule):

    def __init__(self,nhead = 8,nlayers = 1 ,**kwargs):
        super(ItemSageModel,self).__init__()
        
        self.margin = kwargs['loss_margin']
        if self.margin :
            self.loss = kwargs['loss'](margin=kwargs['loss_margin']) # criterion created
        else:
            self.loss = kwargs['loss']( )
            
        self.log_interval = kwargs['log_interval']
        self.loss_type = kwargs['loss_type']
        # self.len_train_data = kwargs['train_size']
        self.accuracy = torchmetrics.Accuracy(task='binary', average='macro')
        self.f1score = torchmetrics.F1Score(task='binary', average='macro')
        self.precision = torchmetrics.Precision(task='binary', average='macro')
        self.recall = torchmetrics.Recall(task='binary', average='macro')
        
        self.home_feat_dim = kwargs['home_features_dim']
        self.image_embedding_dim = kwargs['image_embed_dim']
        self.n_images = kwargs['n_img']
        self.text_embedding_dim = kwargs['text_embed_dim']
        self.skg_embedding_dim = kwargs['skg_embed_dim']
        self.ps_embedding_dim = kwargs['ps_embed_dim']
        self.transformer_input_dim = kwargs['transformer_input_dim']
        self.output_dim = kwargs['final_embed_dim']
        self.n_zips = kwargs['n_zips']
        self.zip_embed_dim = kwargs['zip_embed_dim']
        
        # transformer params
        self.nhead = nhead
        self.nlayers = nlayers
        
        # build model
        self.__model_build()
        
    def __model_build(self) :
        
        # converting all input to the same dimensionality for forming transformer input seq
        # if self.home_features_input:
        self.zip_embed = nn.Embedding(self.n_zips,self.zip_embed_dim)
        self.home_linear = nn.Linear(self.home_feat_dim+self.zip_embed_dim, self.transformer_input_dim)
        # if self.image_input:
        self.image_linear = nn.Linear(self.image_embedding_dim, self.transformer_input_dim)
        # if self.text_input:
        self.text_linear = nn.Linear(self.text_embedding_dim, self.transformer_input_dim)
        # if self.skg_input:
        self.skg_linear = nn.Linear(self.skg_embedding_dim, self.transformer_input_dim)
        # if self.ps_input:
        # self.ps_linear = nn.Linear(self.ps_embedding_dim, self.transformer_input_dim)
        
        # transforming the CLS token
        self.global_CLS_linear = nn.Linear(1, self.transformer_input_dim)
        
        # Transformer Encoder Layer
        encoder_layer = TransformerEncoderLayer(d_model=self.transformer_input_dim, nhead=self.nhead,\
                                                     batch_first=True,norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.nlayers)
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.transformer_input_dim),
            nn.Linear(self.transformer_input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, self.output_dim)
        )
        
    # itemsage tower
    def _encoder(self, text_embeddings, image_embeddings, skg_embeddings, home_features, zip_idx):
        """ 
        :param image_embeddings: (bsz, seq_length, D_img) - 
        :param text_embeddings: (bsz,  D_text) - make seq_length = 1 for openAI embeddings - using unsqueeze()
        :param skg_embeddings: (bsz, D_skg) - make seq_length = 1 because 1 embedding/listing - using unsqueeze()
        :param ps_embeddings: (bsz, D_ps) - make seq_length = 1 because 1 embedding/listing - using unsqueeze()
        :param home_features: (bsz, D_home_feat) - make seq_length = 1 because 1 embedding/listing - using unsqueeze()
        """
        # Apply linear transformation to the global token
        batch_size = image_embeddings.size(0)
        cls_token = torch.ones(batch_size, 1,1, device=self.device)
        
        # Apply linear transformation to make [CLS] token 512-dimensional
        cls_token = self.global_CLS_linear(cls_token)
        
        transformed_image_embeddings = self.image_linear(image_embeddings)
        transformed_text_embeddings = self.text_linear(torch.unsqueeze(text_embeddings,1))
        transformed_skg_embeddings = self.skg_linear(torch.unsqueeze(skg_embeddings,1))
        # transformed_ps_embeddings = self.ps_linear(torch.unsqueeze(ps_embeddings,1))
        zip_embedding = self.zip_embed(zip_idx)
        home_features = torch.cat((home_features,zip_embedding),dim=1)
        transformed_home_feat = self.home_linear(torch.unsqueeze(home_features,1))

        embedding_seq = torch.cat((cls_token, transformed_image_embeddings, transformed_text_embeddings,\
                                   transformed_skg_embeddings,transformed_home_feat), dim=1)
        # print(cls_token.shape, transformed_image_embeddings.shape, transformed_text_embeddings.shape)
        # print('embedding_seq shape : ',embedding_seq.shape)
        transformer_output = self.transformer_encoder(embedding_seq)
        # print('transformer_output shape : ',transformer_output.shape)
        cls_output = transformer_output[:, 0, :]
        product_embedding = self.mlp_head(cls_output)
        return F.normalize(product_embedding,dim=1)
    
    
    def forward(self,**kwargs1):

        anchor_encoder = self._encoder(kwargs1['anchor'][0],kwargs1['anchor'][1],kwargs1['anchor'][2],\
                                      kwargs1['anchor'][3],kwargs1['anchor'][4]) # query tower

        engaged_encoder = self._encoder(kwargs1['engaged'][0],kwargs1['engaged'][1],kwargs1['engaged'][2],\
                                       kwargs1['engaged'][3],kwargs1['engaged'][4]) # example tower
        
        # normalize both embeddings so that cosine_similarity == dot_product
        return anchor_encoder,engaged_encoder
    
        
    # option to use one of the two loss types
    def _contrastive_loss(self, x1,x2,label):
        return self.loss(x1,x2,label)
    
    def _classification_loss(self,score,label):
        return self.loss(score,label)

    def training_step(self, batch,batch_idx):
        kwargs1 = {'anchor':[],'engaged':[]}
        
        #labels
        batch_labels = batch['label']
        bsz = len(batch_labels)
        

        ## tower inputs
        kwargs1['anchor'] = [batch['text_embed1'],
                             batch['img_embed1'].view(bsz,self.n_images,self.image_embedding_dim),
                             batch['skg_embed1'],
                             batch['home_features1'][:,:self.home_feat_dim],
                             batch['home_features1'][:,self.home_feat_dim].to(torch.long)
                            ]
        
        kwargs1['engaged'] = [batch['text_embed2'],
                             batch['img_embed2'].view(bsz,self.n_images,self.image_embedding_dim),
                             batch['skg_embed2'],
                             batch['home_features2'][:,:self.home_feat_dim],
                             batch['home_features2'][:,self.home_feat_dim].to(torch.long)
                            ]
        
        # model forward pass
        q_emb,p_emb = self(**kwargs1)
        
        if self.loss_type ==  "contrastive" : # contrastive
            batch_labels = torch.where(batch_labels == 0,-1,batch_labels) # for CosineEmbeddingloss()
            loss = self._contrastive_loss( q_emb,p_emb,batch_labels )
        else: #classification
            scores = F.sigmoid( F.cosine_similarity( q_emb,p_emb ) ) # applying sigmoid to get score between 0 and 1
            loss = self._classification_loss( scores,batch_labels.to(torch.float32)  )
        
        # get predictions
        batch_labels = torch.where (batch_labels == -1,0,batch_labels)
        if self.loss_type  ==  "contrastive" :
            predicted_prob = F.sigmoid(F.cosine_similarity(q_emb,p_emb) - self.margin)
        else:
            predicted_prob = F.sigmoid(F.cosine_similarity(q_emb,p_emb))
        preds = torch.where (predicted_prob >=0.5,1,0)
        
        # metrics_calculation
        batch_acc = self.accuracy(preds,batch_labels)
        batch_f1 = self.f1score(preds,batch_labels)
        batch_prec = self.precision(preds,batch_labels)
        batch_recall = self.recall(preds,batch_labels)
        log_dict = {
            'train_accuracy' : batch_acc,
            'train_precision' : batch_prec,
            'train_recall' : batch_recall
        }
        
        self.log('train_loss',loss,on_epoch = True,prog_bar=True)
        self.log('train_f1score',batch_f1,on_epoch = True,prog_bar=True)
        
        self.log_dict(log_dict,on_step=False, on_epoch = True)
        
        return loss
    
    def validation_step(self, batch,batch_idx):
        kwargs1 = {'anchor':[],'engaged':[]}
        
        #labels
        batch_labels = batch['label']
        bsz = len(batch_labels)
        
        ## tower inputs
        kwargs1['anchor'] = [batch['text_embed1'],
                             batch['img_embed1'].view(bsz,self.n_images,self.image_embedding_dim),
                             batch['skg_embed1'],
                             batch['home_features1'][:,:self.home_feat_dim],
                             batch['home_features1'][:,self.home_feat_dim].to(torch.long)
                            ]
        
        kwargs1['engaged'] = [batch['text_embed2'],
                             batch['img_embed2'].view(bsz,self.n_images,self.image_embedding_dim),
                             batch['skg_embed2'],
                             batch['home_features2'][:,:self.home_feat_dim],
                             batch['home_features2'][:,self.home_feat_dim].to(torch.long)
                            ]
 
        
        # model forward pass
        q_emb,p_emb = self(**kwargs1)
        
        if self.loss_type ==  "contrastive" : # contrastive
            batch_labels = torch.where(batch_labels == 0,-1,batch_labels) # for CosineEmbeddingloss()
            loss = self._contrastive_loss( q_emb,p_emb,batch_labels ).item()
        else: #classification
            scores = F.sigmoid( F.cosine_similarity( q_emb,p_emb ) ) # applying sigmoid to get score between 0 and 1
            loss = self._classification_loss( scores,batch_labels.to(torch.float32)  ).item()
            
        # get predictions
        batch_labels = torch.where (batch_labels == -1,0,batch_labels)
        if self.loss_type ==  "contrastive" :
            predicted_prob = F.sigmoid(F.cosine_similarity(q_emb,p_emb) - self.margin)
        else:
            predicted_prob = F.sigmoid(F.cosine_similarity(q_emb,p_emb))
        preds = torch.where (predicted_prob >=0.5,1,0)
        
        # metrics_calculation
        batch_acc = self.accuracy(preds,batch_labels)
        batch_f1 = self.f1score(preds,batch_labels)
        batch_prec = self.precision(preds,batch_labels)
        batch_recall = self.recall(preds,batch_labels)
        log_dict = {
            'val_loss': loss,
            'val_accuracy' : batch_acc,
            'val_f1score' : batch_f1,
            'val_precision' : batch_prec,
            'val_recall' : batch_recall
        }
        self.log_dict(log_dict,on_epoch = True, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
        return [optimizer], [lr_scheduler]
    
