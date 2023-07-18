import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from model import ItemSageModel
from data import ItemsageDataModule

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


os.chdir(os.path.join(os.path.expanduser("~"),'ItemSage_Data_30dayLB'))
home_dir = os.getcwd()
TRAIN_PATH = 'train_img_flattened'
# TEST_PATH = 'test_img_flattened'
VAL_PATH = 'val_img_flattened'
# TRAIN_SIZE = 1757020
NUM_EPOCHS = 25
ENGAGEMENT_TYPE = 'all' # clicks/saves/all

LOG_INTERVAL = 1300 # just to keep ~10 reporting logs per epoch
BATCH_SIZE = 1024

HOME_FEATS_DIM = 28
IMAGE_INPUT_EMBED_DIM = 512 
NUM_IMAGES = 5
TEXT_INPUT_EMBED_DIM = 1536 
SKG_INPUT_EMBED_DIM = 32  # skip-gram
PS_INPUT_EMBED_DIM = 128 # pinsage
TRANSFORMER_INPUT_DIM = 512
ITEMSAGE_EMBED_DIM = 256
N_ZIP_CATEGORIES = 14 # look this up everytime while creating data
ZIP_EMBED_DIM = 16
SEED = 42
LOSS_TYPE = "contrastive" # contrastive/classification/itemsage_loss
# LOSS_FUNC = nn.BCELoss # classification : BCELoss() | contrastive : nn.CosineEmbeddingLoss() | itemsage_loss : simCLRLoss() [custom - to be implemented]
CONTRASTIVE_LOSS_MARGIN = 0.1
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main(model_save_dir, log_dir,  **kwargs):

    # init data
    data_module = ItemsageDataModule(**kwargs)
    # init model
    model = ItemSageModel(**kwargs)
    model = torch.compile(model)

    # define callbacks
    checkpoint_callback = ModelCheckpoint(
    dirpath=model_save_dir,
    filename='model-{epoch:02d}',
    save_top_k=-1,  # Save all checkpoints
    verbose=True)

    early_stop_callback = EarlyStopping(
    monitor='val_loss',  # The metric to monitor (e.g., validation loss)
    mode='min',  # The direction to monitor (minimize the metric)
    patience=5,  # Number of epochs with no improvement before stopping
    verbose=True  # Print early stopping updates
    )

    callbacks = [checkpoint_callback,early_stop_callback]

    # init trainer
    trainer = Trainer(
        callbacks= callbacks ,
        max_epochs=kwargs['n_epochs'],
        accelerator="gpu", 
        devices="auto",
        strategy="fsdp_native",
        deterministic=True,
        precision='16-mixed',
        default_root_dir=log_dir,
        reload_dataloaders_every_n_epochs=1
    )
    
    # start training
    trainer.fit( model=model, datamodule=data_module )



if __name__ == '__main__':

    set_seed(SEED)

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
    'train_path':''.join(['file://',os.path.join(home_dir, TRAIN_PATH)]),
    'val_path':''.join(['file://',os.path.join(home_dir, VAL_PATH)]),
    'model_path': os.path.join(home_dir, 'Models')
    }

    model_save_dir = Path(os.path.join(model_args['model_path'],'_'.join(['model',LOSS_TYPE,datetime.now().strftime('%Y-%m-%d_%H-%M-%S')])))
    if not model_save_dir.exists():
        model_save_dir.mkdir(parents=True)
    
    log_file = os.path.join(model_save_dir,'output.log')
    log_dir = Path(os.path.join(model_save_dir,'log'))
    
    main(model_save_dir, log_dir,  **model_args)


