import torch.nn as nn
from Preprocessing import PAD_INDEX


#HYPERPARAMETER TUNING
EMBEDDING_DIM = 128


class LSTM(nn.Module):
    def __init__(self, output_size, len_vocab, num_layers=1, num_hidden=128):

        super(LSTM,self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings = len_vocab,
            embedding_dim = EMBEDDING_DIM,
            padding_idx = PAD_INDEX
        )
        
        self.lstm = nn.LSTM(
            input_size = num_hidden,
            hidden_size = num_hidden,
            num_layers = num_layers,
            batch_first = True,
            dropout = 0.5
        )

        self.fc_out = nn.Linear(num_hidden, output_size)


    def forward(self, input_seq, hidden_in, mem_in):

        input_embs = self.embedding(input_seq)
        output, (hidden_out, mem_out) = self.lstm(input_embs, (hidden_in, mem_in))
        return self.fc_out(output), hidden_out, mem_out
    

#class LSTM(nn.Module):
    #def __init__(self, num_embedding, output_size, num_layers=1, num_hidden=128):
        
    #    super(LSTM,self).__init__()
    #    self.embedding = nn.Embedding(num_embedding, num_hidden)
    #    self.lstm = nn.LSTM(
    #        input_size=num_hidden,
    #        hidden_size=num_hidden,
    #        num_layers=num_layers,
    ##        dropout=0.5
     #   )
     #   self.fc_out = nn.Linear(num_hidden, output_size)

#    def forward(self, input_seq, hidden_in, mem_in):
#
 #       input_embs = self.embedding(input_seq)
  #      output, (hidden_out, mem_out) = self.lstm(input_embs, (hidden_in, mem_in))
   #     return self.fc_out(output), hidden_out, mem_out