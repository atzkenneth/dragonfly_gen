import h5py
import numpy as np
import pandas as pd
import rdkit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from rdkit import Chem
from torch_geometric.nn.aggr import GraphMultisetTransformer
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, Size, Tensor
from torch_geometric.utils.scatter import scatter

if __name__ == "__main__":
    print(f"torch_geometric.__version__: {torch_geometric.__version__}")
    print(f"torch.__version__: {torch.__version__}")
    print(f"rdkit.__version__: {rdkit.__version__}")
