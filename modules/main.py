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