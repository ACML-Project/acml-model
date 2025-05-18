from Create_Datasets import Split_Dataset, Load_Merged_Data, READABLES
from LSTM import LSTM, EMBEDDING_DIM
import numpy as np
import pandas as pd
from Preprocessing import Preprocess_Data, Load_Data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


#HYPERPARMETER TUNING
BATCH_SIZE = 32
HIDDEN_LAYERS = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_LAYERS = 2
DROP_OUT = 0.3


#GET DATA
unprocessed_data = Load_Merged_Data() 
encoded_data, vocab = Load_Data()

labels = torch.tensor(unprocessed_data['label'].values, dtype = torch.long)
inputs = torch.tensor(encoded_data, dtype = torch.long)
dataset = TensorDataset(inputs, labels)
training_data, validation_data, test_data = Split_Dataset(dataset)

   
#device = torch.device(1 if torch.cuda.is_available() else 'cpu')

#lstm = LSTM(
 #   output_size = 2,
  #  len_vocab = len(vocab),
   # num_layers = NUM_LAYERS,
    #num_hidden = HIDDEN_LAYERS
#).to(device)


#optimizer = torch.optim.Adam(lstm_classifier.parameters, lr = LEARNING_RATE)
#loss_fn = nn.CrossEntropyLoss()
#num_params = 0
#for param in lstm_classifier.parameters():
#    num_params += param.flatten().shape[0]


#print(num_params)


#training_loss_logger = []
#test_loss_logger = []
#training_acc_logger = []
#test_acc_logger = []


#pbar = trange(0, NUM_EPOCHS, leave=False, desc="Epoch")
#train_acc = 0
#test_acc = 0


#INDEPENDENT 
#X = dataset.drop('label', axis=1)

#DEPENDENT
#Y = dataset['label']

#print(len(vocab))
#output_file = open("output.txt", "w")
#print(vocab, file=output_file)
#https://www.youtube.com/watch?v=k3_qIfRogyY
#return ' '.join(processed_words)


#KEEP TRACK OF THE LONGEST ARTICLE
    #global MAX_ARTICLE_LEN
    #len_curr = len(processed_text)
    #if len_curr > MAX_ARTICLE_LEN:
    #    MAX_ARTICLE_LEN = len_curr