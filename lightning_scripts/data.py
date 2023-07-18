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
import petastorm
from petastorm.pytorch import DataLoader
from petastorm import make_reader
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import time

import random
import pytorch_lightning as pl


class ItemsageDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.train_path = kwargs['train_path']
        self.val_path = kwargs['val_path']
        self.batch_size = kwargs['bsz']

    def setup(self):
        pass
        
    def train_dataloader(self):
        self.reader_train = make_reader(self.train_path, num_epochs=1, seed=1, shuffle_rows=True)
        return DataLoader(self.reader_train, batch_size=self.batch_size)

    def val_dataloader(self):
        self.reader_val = make_reader(self.val_path, num_epochs=1, seed=1, shuffle_rows=False)
        return DataLoader(self.reader_val, batch_size=self.batch_size)