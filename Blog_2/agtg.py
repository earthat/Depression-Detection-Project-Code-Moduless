import torch
import pickle
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class AGTG_Model(nn.Module):
        def __init__(self, nodes_dim, node_features_dim):
            super().__init__()
            self.E = nodes_dim   # 
            self.F = node_features_dim
            self.P = nn.Parameter(torch.randn(self.E,self.E), requires_grad=True)
            self.Q = nn.Parameter(torch.randn(self.F,self.E), requires_grad=True)
            self.b = nn.Parameter(torch.randn(self.E,1), requires_grad=True)
            self.d = nn.Parameter(torch.randn(1), requires_grad=True)

        def D_Matrix(self, A):
             d = torch.sum(A, 1)
             d_inv_sqrt = d**(-(1/2))
             D_inv_sqrt = torch.diag(d_inv_sqrt)
             return D_inv_sqrt
        
        def forward(self, X, adjacency_matrix):
            I = torch.eye(self.E)
            PX = torch.matmul(self.P, X)
            PXQ = torch.matmul(PX, self.Q)
            A_cor = torch.abs(PXQ+self.b)
            A = nn.functional.relu(A_cor+adjacency_matrix*self.d)
            A_I = A+I
            D = self.D_Matrix(A_I)
            A_norm = torch.matmul(D, torch.matmul(A_I, D))
            return A_norm