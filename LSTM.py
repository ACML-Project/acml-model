import torch.nn as nn
from Preprocessing import PAD_INDEX

class LSTM(nn.Module):
    def __init__(self, dropout, embedding_dim, hidden_size, len_vocab, num_recurrent_layers, output_size):
        super(LSTM, self).__init__()
        
        # CREATES AN EMBEDDING FOR EACH TOKEN
        self.embedding = nn.Embedding(
            num_embeddings=len_vocab,
            embedding_dim=embedding_dim,
            padding_idx=PAD_INDEX
        )
        
        # LSTM LAYER WITH PROPER DROPOUT HANDLING
        self.lstm = nn.LSTM(
            input_size=embedding_dim,  # NEEDS TO MATCH EMBEDDING VECTOR'S DIMENSIONS
            hidden_size=hidden_size,
            num_layers=num_recurrent_layers,
            batch_first=True,
            dropout=dropout if num_recurrent_layers > 1 else 0,  # Dropout only works with >1 layers
            bidirectional=False  # Set to True for bidirectional LSTM if desired
        )
        
        # ADDITIONAL DROPOUT LAYER FOR OUTPUT
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        
        # OUTPUT LINEAR LAYER
        self.output_vals = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden_in, memory_in):
        # GET EMBEDDINGS
        input_embeddings = self.embedding(input)
        
        # PASS THROUGH LSTM
        output, (hidden_out, memory_out) = self.lstm(input_embeddings, (hidden_in, memory_in))
        
        # APPLY DROPOUT TO OUTPUT BEFORE CLASSIFICATION
        output_dropped = self.dropout(output)
        
        # GET FINAL PREDICTIONS
        return self.output_vals(output_dropped), hidden_out, memory_out