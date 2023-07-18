import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from pyarrow.parquet import ParquetFile
import pyarrow.parquet as pq
# import s3fs
# s3_filesystem = s3fs.S3FileSystem()
import glob
import pandas as pd
import numpy as np
import boto3
import ast
import time
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from random import sample
from tqdm import tqdm
import petastorm
from petastorm.pytorch import DataLoader
from petastorm import make_reader
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import time
import logging

from model import ItemSageModel
from metrics import AccumulatedAccuracyMetric,AccumulatedF1Metric

os.chdir(os.path.join(os.path.expanduser("~"),'ItemSage_Data_v3'))
home_dir = os.getcwd()
TRAIN_PATH = 'train3_img_flattened'
# TEST_PATH = 'test_img_flattened'
VAL_PATH = 'val3_img_flattened'
TRAIN_SIZE = 1757020
NUM_EPOCHS = 25
ENGAGEMENT_TYPE = 'all' # clicks/saves/all
LOG_INTERVAL = 1300 # just to keep ~10 reporting logs per epoch
BATCH_SIZE = 128
HOME_FEATS_DIM = 30
IMAGE_INPUT_EMBED_DIM = 512 
NUM_IMAGES = 10
TEXT_INPUT_EMBED_DIM = 1536 
SKG_INPUT_EMBED_DIM = 32  # skip-gram
PS_INPUT_EMBED_DIM = 128 # pinsage
TRANSFORMER_INPUT_DIM = 512
ITEMSAGE_EMBED_DIM = 256
N_ZIP_CATEGORIES = 14 # look this up everytime while creating data
ZIP_EMBED_DIM = 16

LOSS_TYPE = "contrastive" # contrastive/classification/itemsage_loss
# LOSS_FUNC = nn.BCELoss # classification : BCELoss() | contrastive : nn.CosineEmbeddingLoss() | itemsage_loss : simCLRLoss() [custom - to be implemented]
CONTRASTIVE_LOSS_MARGIN = 0.1
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model_args = {
    'n_epochs' : NUM_EPOCHS,
    'log_interval' : LOG_INTERVAL,
    'bsz':BATCH_SIZE,
    'engagement_type' : ENGAGEMENT_TYPE,
    'loss_type': LOSS_TYPE,
    'loss' : nn.BCELoss if LOSS_TYPE == 'classification' else nn.CosineEmbeddingLoss,
    'loss_margin':None if LOSS_TYPE == 'classification' else CONTRASTIVE_LOSS_MARGIN,
    'home_features_dim' : HOME_FEATS_DIM,
    'image_embed_dim' : IMAGE_INPUT_EMBED_DIM,
    'n_img' : NUM_IMAGES,\
    'text_embed_dim' : TEXT_INPUT_EMBED_DIM,
    'skg_embed_dim' : SKG_INPUT_EMBED_DIM,
    'ps_embed_dim' : PS_INPUT_EMBED_DIM,
    'transformer_input_dim' : TRANSFORMER_INPUT_DIM,
    'final_embed_dim' : ITEMSAGE_EMBED_DIM,
    'n_zips':N_ZIP_CATEGORIES,
    'zip_embed_dim' : ZIP_EMBED_DIM,
    'device' : DEVICE,
    'train_path':''.join(['file://',os.path.join(home_dir, TRAIN_PATH)]),
    # 'test_path':''.join(['file://',os.path.join(home_dir, TEST_PATH)]),
    'val_path':''.join(['file://',os.path.join(home_dir, VAL_PATH)]),
    'model_path': os.path.join(home_dir, 'Models'),
    'train_size' : TRAIN_SIZE
}

cuda_args = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

model_args = {**model_args, **cuda_args}

model_save_dir = Path(os.path.join(model_args['model_path'],'_'.join(['model',LOSS_TYPE,datetime.now().strftime('%Y-%m-%d_%H-%M-%S')])))
if not model_save_dir.exists():
        model_save_dir.mkdir(parents=True)
        
log_file = os.path.join(model_save_dir,'output.log')
logging.basicConfig(filename=log_file, level=logging.INFO)



model_args['metrics'] = [AccumulatedAccuracyMetric(model_args['loss_margin']),AccumulatedF1Metric(model_args['loss_margin'])]

def main(model_args = model_args):

    model = ItemSageModel(**model_args).to(model_args['device'])
    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    early_stopping_patience = 5  # Number of epochs without improvement before stopping
    best_validation_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = -1
    best_accuracy = 0.

    if model_args['loss_margin'] :
        logging.info(" ###### USING CONTRASTIVE LOSS ######\n ")
        print( " ###### USING CONTRASTIVE LOSS ######\n ")
    else :
        logging.info( " ###### USING CLASSIFICATION LOSS ######\n ")
        print( " ###### USING CLASSIFICATION LOSS ######\n ")
        
    for epoch in range(1,model_args['n_epochs']+1 ):
        scheduler.step()
        tic = time.time()
        with DataLoader(make_reader(model_args['train_path'], num_epochs=1, seed=1, shuffle_rows=True),
                        batch_size=model_args['bsz']) as train_loader:
            epoch_train_loss,metrics = train_one_epoch(epoch,train_loader,model,optimizer,**model_args)
        message = '\nEpoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch , model_args['n_epochs'], epoch_train_loss)
        
        for metric in metrics:
            message += '\t{}: {:.2f}'.format(metric.name(), metric.value())
        
        with DataLoader(make_reader(model_args['val_path'], num_epochs=1, seed=1, shuffle_rows=True),
                        batch_size=model_args['bsz']) as val_loader:
            epoch_val_loss,metrics = test_one_epoch(epoch,val_loader,model,optimizer,**model_args)
        message += '\nEpoch: {}/{:.2f}. Validation set: Average loss: {:.4f}'.format(epoch , model_args['n_epochs'],epoch_val_loss)
        
        for metric in metrics:
            message += '\t{}: {:.2f}'.format(metric.name(), metric.value())
        
        # save epoch model states
        model_filename = 'model_state_epoch_{}.pth'.format(epoch)
        torch.save(model.state_dict(), os.path.join(model_save_dir,model_filename))
        toc = time.time()
        message += '\nEpoch time : {} min. Model state saved to : {}'.format(np.round((toc-tic)/60,2),os.path.join(model_save_dir,model_filename))
        
        logging.info(message)
        print(message)
        
        # Early Stopping based on train_loss - val_loss and val accuracy
        acc = metrics[0].value()
        # epoch_loss_diff = abs(epoch_train_loss - epoch_val_loss)
        # if (epoch_loss_diff < min_diff_till_now) & (epoch_val_acc >= best_val_acc-1):
        #     min_diff_till_now = epoch_loss_diff
        #     best_val_acc = epoch_val_acc
        #     epochs_without_improvement = 0
        # else:
        #     epochs_without_improvement += 1
            
        if acc > best_accuracy:
            best_accuracy = acc
            best_epoch = epoch
        elif epoch - best_epoch > early_stopping_patience:
            logging.info("Early stopping triggered. No improvement in validation Accuracy. Best Epoch : {}".format(best_epoch))
            print("Early stopping triggered. No improvement in validation Accuracy. Best Epoch : {}".format(best_epoch))
            break  # terminate the training loop
            
        # if epoch_val_loss < best_validation_loss:
        #     best_validation_loss = epoch_val_loss
        #     epochs_without_improvement = 0
        # else:
        #     epochs_without_improvement += 1
        
        # if epochs_without_improvement >= early_stopping_patience:
        #     logging.info("Early stopping triggered. No improvement in validation loss.")
        #     print("Early stopping triggered. No improvement in validation loss.")
        #     break

def train_one_epoch(epoch,train_loader,model,optimizer,**kwargs):
    # model = kwargs['model']
    model.train()
    
    log_interval = kwargs['log_interval']
    # bsz = kwargs['bsz']
    device = kwargs['device']
    loss_type = kwargs['loss_type']
    metrics = kwargs['metrics']
    len_train_data = kwargs['train_size']
    for metric in metrics:
        metric.reset()
    train_loss = 0.
    losses = []
    
    # progress_bar = tqdm(total=BATCH_SIZE_POS, desc='Training Progress')
    
    counter = 0
    for batch_idx, batch in enumerate(train_loader):
        kwargs1 = {'anchor':[],'engaged':[]}
        
        #labels
        batch_labels = batch['label'].to(device)
        bsz = len(batch_labels)
        
        ## anchor tower input
        kwargs1['anchor']= [batch['text_embed1'].to(device),\
                            batch['img_embed1'].view(bsz,kwargs['n_img'],kwargs['image_embed_dim']).to(device),\
                            batch['skg_embed1'].to(device),\
                            batch['ps_embed1'].to(device),\
                            batch['home_features1'][:,:kwargs['home_features_dim']].to(device),\
                            batch['home_features1'][:,kwargs['home_features_dim']].to(torch.long).to(device)]
        ## engaged tower input
        kwargs1['engaged']= [batch['text_embed2'].to(device),\
                             batch['img_embed2'].view(bsz,kwargs['n_img'],kwargs['image_embed_dim']).to(device),\
                             batch['skg_embed2'].to(device),\
                             batch['ps_embed2'].to(device),\
                             batch['home_features2'][:,:kwargs['home_features_dim']].to(device),\
                             batch['home_features2'][:,kwargs['home_features_dim']].to(torch.long).to(device)]
        # model forward pass
        q_emb,p_emb = model(**kwargs1)
        
        # optimizer 
        optimizer.zero_grad()
        
        if loss_type ==  "contrastive" : # contrastive
            batch_labels = torch.where(batch_labels == 0,-1,batch_labels) # for CosineEmbeddingloss()
            loss = model._contrastive_loss( q_emb,p_emb,batch_labels )
        else: #classification
            scores = F.sigmoid( F.cosine_similarity( q_emb,p_emb ) ) # applying sigmoid to get score between 0 and 1
            loss = model._classification_loss( scores,batch_labels.to(torch.float32)  )
            
        # model backward pass
        loss.backward()
        # wt updation
        optimizer.step()
        train_loss += loss.item()
        losses.append(loss.item())
        counter += bsz
        
        for metric in metrics:
            metric(q_emb,p_emb, batch_labels)
        
        if batch_idx % log_interval == 0:
            message = 'Train : {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, counter,len_train_data,100.*counter/len_train_data, np.round(np.mean(losses),4)) # np.round(train_loss/counter,4)
            for metric in metrics:
                message += '\t{}: {:.2f}'.format(metric.name(), metric.value())
            
            logging.info(message)
            print(message)
            losses = []
            
    train_loss /= (batch_idx+1)
    return train_loss,metrics # avg train epoch loss - averaged across all batches
    
def test_one_epoch(epoch,test_loader,model,optimizer,val=True,**kwargs):
    
    metrics = kwargs['metrics']
    with torch.no_grad():
        # metrics = kwargs['metrics']
        for metric in metrics:
            metric.reset()
            
    # model = kwargs['model']
    model.eval()
    
    log_interval = kwargs['log_interval']
    # bsz = kwargs['bsz']
    device = kwargs['device']
    loss_type = kwargs['loss_type']
    
    test_loss = 0
    correct = 0
    count = 0
    
    with torch.no_grad():
        for batch_idx,batch in enumerate(test_loader):
            kwargs1 = {'anchor':[],'engaged':[]}
            
            #labels
            batch_labels = batch['label'].to(device)
            bsz = len(batch_labels)
            
            ## anchor tower input
            kwargs1['anchor']= [batch['text_embed1'].to(device),\
                                batch['img_embed1'].view(bsz,kwargs['n_img'],kwargs['image_embed_dim']).to(device),\
                                batch['skg_embed1'].to(device),\
                                batch['ps_embed1'].to(device),\
                                batch['home_features1'][:,:kwargs['home_features_dim']].to(device),\
                                batch['home_features1'][:,kwargs['home_features_dim']].to(torch.long).to(device)]
            ## engaged tower input
            kwargs1['engaged']= [batch['text_embed2'].to(device),\
                                 batch['img_embed2'].view(bsz,kwargs['n_img'],kwargs['image_embed_dim']).to(device),\
                                 batch['skg_embed2'].to(device),\
                                 batch['ps_embed2'].to(device),\
                                 batch['home_features2'][:,:kwargs['home_features_dim']].to(device),\
                                 batch['home_features2'][:,kwargs['home_features_dim']].to(torch.long).to(device)]

            # model forward pass
            q_emb,p_emb = model(**kwargs1)
            
            
            if loss_type ==  "contrastive" : # contrastive
                batch_labels = torch.where(batch_labels == 0,-1,batch_labels) # for CosineEmbeddingloss()
                loss = model._contrastive_loss( q_emb,p_emb,batch_labels )
            else: #classification
                scores = F.sigmoid( F.cosine_similarity( q_emb,p_emb ) ) # applying sigmoid to get score between 0 and 1
                loss = model._classification_loss( scores,batch_labels.to(torch.float32) )
            
            test_loss += loss.item()  # sum up batch loss
            count += bsz
            for metric in metrics:
                metric(q_emb,p_emb, batch_labels)
    test_loss /= (batch_idx+1)
    return test_loss,metrics
    

if __name__ == "__main__"   :
    main()
    
  
