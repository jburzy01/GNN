from torch_geometric.nn import GCNConv,GATv2Conv 
import torch.nn as nn
import torch.nn.functional as F


class GNNVertexFinder(nn.Module):
    """
    Vertex finding model, which takes a pair of nodes in the graph and 
    outputs a binary label to say whether these nodes are in the same vertex.
    """
    def __init__(self, input_size, output_size):
        super().__init__()

        # node classification
        self.conv1 = GCNConv(input_size, 16)
        self.conv2 = GCNConv(16, output_size)


    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)