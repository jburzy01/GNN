import pytorch_lightning as pl
import torch.distributed as dist
from torch_geometric.loader import DataLoader

from data.dataset import TrackDataset

class TrackDataModule(pl.LightningDataModule):

    def __init__(self):

        self.path = "/Users/jburzyns/Documents/work/LLP/HDMI/GNN/gnn_vertex_finder/training/data/" 
        self.batch_size = 1
        self.num_workers = 8

        self.prepare_data_per_node = True


    def setup(self, stage=None):

        self.dataset = TrackDataset(root=self.path)
        self.dataset = self.dataset.shuffle()

        num_samples = self.dataset.len()
        num_val = num_samples // 10

        self.val_dset = self.dataset[:num_val]
        self.test_dset = self.dataset[num_val:2 * num_val]
        self.train_dset = self.dataset[2 * num_val:]

    def train_dataloader(self):

        # create dataloader
        return DataLoader(
            self.train_dset,
            self.batch_size,
            num_workers=self.num_workers
        )


    def val_dataloader(self):

        # create dataloader
        return DataLoader(
            self.val_dset,
            self.batch_size,
            num_workers=1,
        )

    def test_dataloader(self):

        # create dataloader
        return DataLoader(
            self.test_dset,
            self.batch_size,
            num_workers=1,
        )
