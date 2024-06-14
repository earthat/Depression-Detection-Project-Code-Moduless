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