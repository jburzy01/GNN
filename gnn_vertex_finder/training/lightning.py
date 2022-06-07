"""
    pytorch lightning LightningModule - the top level model file for the GNN tagger.
"""

import torch
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
        self.model = GNNVertexFinder(input_size = 20, output_size = 1)

        self.lr = self.learning_rate = 0.001



    def forward(self, data):

        # compute the model output given an input graph
        x_out = self.model(data)
        return x_out

	
    def training_step(self, batch, batch_idx):

        data = batch

        x_out = self.forward(data)

        loss = F.binary_cross_entropy(x_out, data.y.unsqueeze(1))

        # metrics here
        pred = x_out.argmax(-1)
        label = data.y
        accuracy = (pred == label).sum() / pred.shape[0]

        self.log("loss/train", loss)
        self.log("accuracy/train", accuracy)

        return loss

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """

        # optimise the whole model
        return torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))

