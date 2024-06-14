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