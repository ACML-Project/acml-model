learning_rate = 1
num_epochs = 20
batch_size = 32
max_len = 128


dataset = 0

#INDEPENDENT 
X = dataset.drop('label', axis=1)

#DEPENDENT
Y = dataset['label']

