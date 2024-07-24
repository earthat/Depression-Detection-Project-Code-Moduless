# gtmp.py

import torch
from torch import nn
from gcngru import GRU_Model  # Import GRU_Model from gcngru.py

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
        AHW_b = torch.matmul(torch.matmul(A, H), self.W) + self.b
        S_node = torch.relu(AHW_b)
        N_idx = torch.argmax(S_node, dim=0)
        mask_n_idx = torch.zeros(self.E, N_idx.shape[0]).scatter_(0, N_idx.unsqueeze(0), 1.)
        V_graph = torch.mul(mask_n_idx, H)
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
