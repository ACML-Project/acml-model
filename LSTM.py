import torch.nn as nn
from Preprocessing import PAD_INDEX


class LSTM(nn.Module):
    def __init__(self, dropout, embedding_dim, hidden_size, len_vocab, num_recurrent_layers, output_size):
        super(LSTM,self).__init__()

        #CREATES AN EMBEDDING FOR EACH TOKEN
        self.embedding = nn.Embedding(
            embedding_dim = embedding_dim,
            num_embeddings = len_vocab,
            padding_idx = PAD_INDEX
        )
        
        self.lstm = nn.LSTM(
            batch_first = True,
            dropout = dropout,
            hidden_size = hidden_size,
            input_size = embedding_dim, #NEEDS TO MATCH EMBEDDING VECTOR'S DIMENSIONS
            num_layers = num_recurrent_layers 
        )

        self.output_layer = nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden_in, memory_in):

        input_embeddings = self.embedding(input)
        output, (hidden_out, memory_out) = self.lstm(input_embeddings, (hidden_in, memory_in))
        return self.output_layer(output), hidden_out, memory_out