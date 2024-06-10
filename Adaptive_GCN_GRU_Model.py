# %%
import pickle
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# %%
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

# %%
class GCN_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.W)
        self.dropout_prob = dropout_prob

    def forward(self, x, adjacency_matrix):
        # Compute the input to the layer: ˆAXW
        AXW = torch.matmul(adjacency_matrix, torch.matmul(x, self.W))

        # Apply the ReLU activation function to the input of the layer: ReLU(ˆAXW)
        hidden_rep = torch.relu(AXW)

        # Apply dropout
        hidden_rep = nn.functional.dropout(hidden_rep, p=self.dropout_prob, training=self.training)

        return hidden_rep

# %%
class GCN_Model(nn.Module):
    def __init__(self, node_features_dim, hidden_dim, num_layers=2, dropout_prob=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        
        self.gcn_layers = nn.ModuleList([self._create_gcn_layer(node_features_dim, hidden_dim) if i == 0
                                         else self._create_gcn_layer(hidden_dim, hidden_dim)
                                         for i in range(self.num_layers)])

    def _create_gcn_layer(self, input_dim, output_dim):
        return GCN_Layer(input_dim, output_dim, dropout_prob=self.dropout_prob)

    def forward(self, features, adaptive_adjacency_matrix):
        x = features
        layers = []
        for i in range(self.num_layers):
            assert self.num_layers==2
            x = self.gcn_layers[i](x, adaptive_adjacency_matrix)
            layers.append(x)
            y = torch.concat(layers, dim=1)
        return y

# %%
class GRU_Cell(nn.Module):
    def __init__(self, nodes_dim):
        E = nodes_dim
        super().__init__()
        
        self.Wr_x = nn.Parameter(torch.randn(E,E), requires_grad=True)
        self.Wr_h = nn.Parameter(torch.randn(E,E), requires_grad=True)
        self.br   = nn.Parameter(torch.randn(E,1), requires_grad=True)

        self.Wu_x = nn.Parameter(torch.randn(E,E), requires_grad=True)
        self.Wu_h = nn.Parameter(torch.randn(E,E), requires_grad=True)
        self.bu   = nn.Parameter(torch.randn(E,1), requires_grad=True)

        self.Wc_x = nn.Parameter(torch.randn(E,E), requires_grad=True)
        self.Wc_h = nn.Parameter(torch.randn(E,E), requires_grad=True)
        self.bc   = nn.Parameter(torch.randn(E,1), requires_grad=True)

    def forward(self, features_t_0, hidden_state_t_1):
        GCN_x_t_0, h_t_1 = features_t_0, hidden_state_t_1
        r_t_0 = nn.functional.sigmoid(torch.matmul(self.Wr_x, GCN_x_t_0)+torch.matmul(self.Wr_h, h_t_1)+self.br)
        u_t_0 = nn.functional.sigmoid(torch.matmul(self.Wu_x, GCN_x_t_0)+torch.matmul(self.Wu_h, h_t_1)+self.bu)
        c_t_0 = nn.functional.tanh(torch.matmul(self.Wc_x, GCN_x_t_0)+torch.matmul(self.Wc_h, torch.mul(r_t_0, h_t_1))+self.bc)
        h_t_0 = torch.mul(u_t_0, h_t_1)+torch.mul(1-u_t_0, c_t_0)
        return h_t_0

# %%
class GRU_Model(nn.Module):
    def __init__(self, nodes_Dim, node_Features_Dim, hidden_dim, num_layers, dropout_prob, Seq_len):
        super().__init__()
        self.Seq_len = Seq_len
        self.agtg_Adj_Matrix_layer = self._create_AGTG_Adj_Matrix(nodes_Dim, node_Features_Dim)
        self.gcn_model_layer = self._create_GCN_Model(node_Features_Dim, hidden_dim, num_layers, dropout_prob)
        self.gru_cell_layer = self._create_GRU_Cell(nodes_Dim)
        self.H_0 = torch.zeros(nodes_Dim, 2*hidden_dim)

    def _create_AGTG_Adj_Matrix(self, nodes_dim, node_features_dim):
        return AGTG_Model(nodes_dim, node_features_dim)
    
    def _create_GCN_Model(self, node_features_dim, hidden_dim, num_layers, dropout_prob):
        return GCN_Model(node_features_dim, hidden_dim, num_layers, dropout_prob)
    
    def _create_GRU_Cell(self, nodes_dim):
        return GRU_Cell(nodes_dim)

    def forward(self, input_seq, adjacency_matrix):
        x, adj = input_seq, adjacency_matrix
        steps, _, _ = x.shape
        assert self.Seq_len == steps
        
        H = self.H_0
        for i in range(self.Seq_len):
            adaptive_adj_i = self.agtg_Adj_Matrix_layer(x[i], adj)
            gcn_Gi = self.gcn_model_layer(x[i], adaptive_adj_i)
            gru_Hi = self.gru_cell_layer(gcn_Gi, H)
            H = gru_Hi  
        return H, adaptive_adj_i

# %%
class GraphTopologyMaxPooling_Model(nn.Module):
    def __init__(self, nodes_Dim, node_Features_Dim, hidden_dim, num_layers, dropout_prob, Seq_len):
        super().__init__()
        self.E = nodes_Dim
        self.W = nn.Parameter(torch.randn(2*hidden_dim, 2*hidden_dim), requires_grad=True)
        self.b = nn.Parameter(torch.randn(nodes_Dim, 1), requires_grad=True)
        self.W_logit = nn.Parameter(torch.randn(2*hidden_dim), requires_grad=True)
        self.gru_Model = self._create_GRU_Model(nodes_Dim, node_Features_Dim, hidden_dim, num_layers, dropout_prob, Seq_len)

    def _create_GRU_Model(self, nodes_Dim, node_Features_Dim, hidden_dim, num_layers, dropout_prob, Seq_len):
        return GRU_Model(nodes_Dim, node_Features_Dim, hidden_dim, num_layers, dropout_prob, Seq_len)

    def forward(self, input_seq, adjacency_matrix):
        H, A = self.gru_Model(input_seq, adjacency_matrix)
        AHW_b = torch.matmul(torch.matmul(A, H), self.W)+self.b
        S_node = torch.relu(AHW_b)
        N_idx = torch.argmax(S_node, dim=0)
        mask__n_idx = torch.zeros(self.E, N_idx.shape[0]).scatter_(0, N_idx.unsqueeze(0), 1.)
        V_graph = torch.mul(mask__n_idx, H)
        logit = torch.matmul(torch.sum(V_graph, 0), self.W_logit)
        return logit

# %%
class Main_Model(nn.Module):
    
    def __init__(self, nodes_Dim, node_Features_Dim, hidden_dim, num_layers, dropout_prob, Seq_len, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.maxPooling_Model = self._create_MaxPooling_Model(nodes_Dim, node_Features_Dim, hidden_dim, num_layers, dropout_prob, Seq_len)

    def _create_MaxPooling_Model(self, nodes_Dim, node_Features_Dim, hidden_dim, num_layers, dropout_prob, Seq_len):
        return GraphTopologyMaxPooling_Model(nodes_Dim, node_Features_Dim, hidden_dim, num_layers, dropout_prob, Seq_len)

    def forward(self, input_seq_batch, adjacency_matrix_batch):
        logits = []
        if input_seq_batch.shape[0] == self.batch_size:
                assert input_seq_batch.shape[0] == self.batch_size
                assert adjacency_matrix_batch.shape[0] == self.batch_size
        else:
            self.batch_size = input_seq_batch.shape[0]

        for i in range(self.batch_size):
            logit = self.maxPooling_Model(input_seq_batch[i], adjacency_matrix_batch[i])
            logits.append(logit)
        logit_vector = torch.stack(logits, dim=0)

        return logit_vector