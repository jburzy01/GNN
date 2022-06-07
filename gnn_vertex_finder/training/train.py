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
from data.datamodule import TrackDataModule


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

    # create a new model
    model = VertexFinder()

    # create the datamodule
    dm = TrackDataModule()

    num_epochs = 10 

    # create the lightning trainer
    print('Creating trainer...')
    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        #strategy=DDPPlugin(find_unused_parameters=False), 
        #accelerator=config['accelerator'],
        #devices=config['num_gpus'],
        logger=None, 
        #log_every_n_steps=20,
        #fast_dev_run=args.test_run,
        #callbacks=callbacks,
    )

    # fit model 
    print('Fitting model...')
    trainer.fit(model, datamodule=dm)

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
