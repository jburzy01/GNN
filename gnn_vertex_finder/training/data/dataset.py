import torch
from torch_geometric.data import Dataset, Data

import uproot
import numpy as np

import os 

class TrackDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return 'GNNTrackHists.root'

    @property
    def processed_file_names(self):
        return 'not_implemented.pt' 

    def download(self):
        pass

    def process(self):
        with uproot.open(self.raw_paths[0] + ':analysis') as tree:
            events = tree.arrays(["track_d0",
                                  "track_z0",
                                  "track_phi0",
                                  "track_theta",
                                  "track_qOverP",
                                  "track_cov00",
                                  "track_cov01",
                                  "track_cov02",
                                  "track_cov03",
                                  "track_cov04",
                                  "track_cov11",
                                  "track_cov12",
                                  "track_cov13",
                                  "track_cov14",
                                  "track_cov22",
                                  "track_cov23",
                                  "track_cov24",
                                  "track_cov33",
                                  "track_cov34",
                                  "track_cov44",
                                  "track_truthIndex"], 
                                  library="np")

            # event-level loop
            for evt in range(len(events['track_d0'])):
                tracks = np.array([events['track_d0'][evt],
                                   events['track_z0'][evt],
                                   events['track_phi0'][evt],
                                   events['track_theta'][evt],
                                   events['track_qOverP'][evt],
                                   events['track_cov00'][evt],
                                   events['track_cov01'][evt],
                                   events['track_cov02'][evt],
                                   events['track_cov03'][evt],
                                   events['track_cov04'][evt],
                                   events['track_cov11'][evt],
                                   events['track_cov12'][evt],
                                   events['track_cov13'][evt],
                                   events['track_cov14'][evt],
                                   events['track_cov22'][evt],
                                   events['track_cov23'][evt],
                                   events['track_cov24'][evt],
                                   events['track_cov33'][evt],
                                   events['track_cov34'][evt],
                                   events['track_cov44'][evt],
                                   events['track_truthIndex'][evt]
                                ])
                tracks = tracks.T

                # TODO: make covariance matrix a single parameter?

                node_features = torch.tensor(tracks[:,:-1], dtype = torch.float)

                adjacency_matrix = np.zeros( (len(tracks), len(tracks)) )

                for i in range(len(tracks)):
                    for j in range(len(tracks)):
                        if i == j:
                            continue
                        if tracks[i][-1] == 0 or tracks[i][-1] == -1:
                            continue

                        if tracks[i][-1] == tracks[j][-1]:
                            adjacency_matrix[i,j] = 1

                row, col = np.where(adjacency_matrix)
                
                coo = np.array(list(zip(row, col)))
                coo = np.reshape(coo, (2,-1))

                adjacency_info = torch.tensor(coo, dtype=torch.long)

                data = Data(x=node_features,
                            edge_index=adjacency_info)

                torch.save(data,os.path.join(self.processed_dir, f'data_{evt}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data