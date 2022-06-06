from torch_geometric.nn import GATv2Conv
import torch.nn as nn




class GNNVertexFinder(nn.Module):
    """
    Vertex finding model, which takes a pair of nodes in the graph and 
    outputs a binary label to say whether these nodes are in the same vertex.
    """
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()

    def forward(self, g):
        return g
