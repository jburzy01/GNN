import sys
import os
import yaml
import glob
import argparse
import shutil
import socket
from datetime import datetime

import comet_ml # have to import before torch/dgl
from pytorch_lightning.loggers import CometLogger

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import TQDMProgressBar as ProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging

from lightning import VertexFinder
from data.dataset import TrackDataset


def parse_args():
    """
    Argument parser for training script.
    """
    parser = argparse.ArgumentParser(description="Train the GNN vertex finder.")
    
    args = parser.parse_args()
    return args


def train(args):
    """
    Fit the model.
    """

    dataset = TrackDataset(root="/Users/jburzyns/Documents/work/LLP/HDMI/GNN/gnn_vertex_finder/training/data/")

    # shuffle dataset and get train/validation/test splits
    dataset = dataset.shuffle()

    num_samples = len(dataset)
    batch_size = 32

    num_val = num_samples // 10

    val_dataset = dataset[:num_val]
    test_dataset = dataset[num_val:2 * num_val]
    train_dataset = dataset[2 * num_val:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    # create a new model
    model = VertexFinder()

    num_epochs = 2500
    val_check_interval = len(train_loader)

    # create the lightning trainer
    print('Creating trainer...')
    trainer = pl.Trainer(max_epochs = num_epochs, \
            val_check_interval=val_check_interval)

    # fit model 
    print('Fitting model...')
    trainer.fit(model, train_loader, val_loader)

    return model, trainer

def main():
    """
    Training entry point.
    """

    # parse args
    args = parse_args()

    # run training
    model, trainer = train(args)


if __name__ == "__main__":
    main()
