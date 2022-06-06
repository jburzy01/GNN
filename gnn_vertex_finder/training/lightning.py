"""
    pytorch lightning LightningModule - the top level model file for the GNN tagger.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl

# model
from models.vertex_finder import GNNVertexFinder

class VertexFinder(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()

        # configuration for this model
        self.config = config

        # create the network
        self.model = GNNVertexFinder(input_size = 516, output_size = 1, hidden_layers = 2)

        self.lr = self.learning_rate = 0.001



    def forward(self, g):

        # compute the model output given an input graph
        return self.model(g)
	
    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x) 

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """

        # optimise the whole model
        return torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))

