from Create_Datasets import Split_Dataset, Load_Merged_Data
from LSTM import LSTM
from Preprocessing import Load_Data, Create_Readable_Text
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from Graph import plot_training

#HYPERPARMETER TUNING
BATCH_SIZE = 32 #number of training samples used in forward/backward pass through the network
DROP_OUT = 0.5 #randomly disregard nodes
EMBEDDING_DIM = 128 #SIZE OF THE VECTOR FOR EACH EMBEDDING
HIDDEN_SIZE = 128 #NUMBER OF FEATURES FOR THE HIDDEN STATE
LEARNING_RATE = 0.005
NUM_EPOCHS = 10
NUM_RECURRENT_LAYERS = 1 #CREATES A STACKED LSTM IF >1. 


#GET DATA FROM PICKLE JAR
unprocessed_data = Load_Merged_Data() 
encoded_data, vocab = Load_Data()

#IF YOU WANT TO READ DATA
#Create_Readable_Text(unprocessed=unprocessed_data, encoding=encoded_data, vocab=vocab)

#CREATES TENSORS FOR THE BINARY LABELS (FAKE/REAL NEWS) AND FOR THE ENCODED TEXT
labels = torch.tensor(unprocessed_data['label'].values, dtype = torch.long)
inputs = torch.tensor(encoded_data, dtype = torch.long)

dataset = TensorDataset(inputs, labels)
training_data, validation_data, test_data = Split_Dataset(dataset)

#if a graphics card is available, we use that. otherwise, stick to cpu.
# device = torch.device(1 if torch.cuda.is_available() else 'cpu')
device = torch.device(0 if torch.cuda.is_available() else 'cpu')

#DATALOADERS MAKE IT EASIER TO LOAD AND PROCESS LARGE DATASETS IN PARALLEL
training_loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)
training_loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(dataset=validation_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

#initialise LSTM
lstm = LSTM(
    dropout = None, #DROP_OUT, CHANGE IN LSTM.PY
    embedding_dim = EMBEDDING_DIM,
    hidden_size = HIDDEN_SIZE,
    num_recurrent_layers = NUM_RECURRENT_LAYERS,
    output_size = 2, #BECAUSE IT'S A BINARY CLASSIFICATION
    len_vocab = len(vocab),
).to(device)

nn.Module.compile(lstm) #SHOULD MAKE COMPUTATION FASTER
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr = LEARNING_RATE)
best_accuracy = 0

#for graphing
train_loss_list = []
val_loss_list = []
train_acc_list =  []
val_acc_list =    []

#TRAINING AND VALIDATION     
for epoch in range(NUM_EPOCHS):

    lstm.train()
    training_loss = 0
    training_correct = 0
    training_total = 0
    
    for inputs, labels in training_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        #get input layer dimension, set to buffer size
        buffer_size = inputs.size(0)
        
        hidden_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
        cell_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
        
        #FOWARD PASS
        optimizer.zero_grad()
        outputs, hidden, memory = lstm(inputs, hidden_state, cell_state)
        outputs = outputs[:, -1, :]  #TAKE THE LAST OUTPUT FOR CLASSIFICATION
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #METRIC CALCULATIONS
        training_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        training_total += labels.size(0)

        #HOW MANY OF THE LSTM'S PREDICTIONS MATCH THE TRUE LABELS FOR FAKE/REAL NEWS
        training_correct += (predicted == labels).sum().item()
    
    #VALIDATION PHASE
    lstm.eval()
    validation_loss = 0
    validation_correct = 0
    validation_total = 0

    #FOR CLASSIFICATION REPORT
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in validation_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            buffer_size = inputs.size(0)
            
            hidden_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
            cell_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
            
            outputs, hidden, memory = lstm(inputs, hidden_state, cell_state)
            outputs = outputs[:, -1, :] #TAKE THE LAST OUTPUT FOR CLASSIFICATION
            
            loss = loss_fn(outputs, labels)
            validation_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            validation_total += labels.size(0)

            #HOW MANY OF THE LSTM'S PREDICTIONS MATCH THE TRUE LABELS FOR FAKE/REAL NEWS
            validation_correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    #METRICS FOR EPOCH
    training_loss = training_loss/len(training_loader)
    training_accuracy = 100*training_correct/training_total
    validation_loss = validation_loss/len(validation_loader)
    validation_accuracy = 100*validation_correct/validation_total

    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}:')
    print(f'Train Loss: {training_loss:.4f} | Acc: {training_accuracy:.2f}%')
    print(f'Val Loss: {validation_loss:.4f} | Acc: {validation_accuracy:.2f}%\n')
    
    #saving results for graphing
    train_loss_list.append(round(training_loss,4))
    val_loss_list.append(round(validation_loss,4))
    train_acc_list.append(round(training_accuracy,4))
    val_acc_list.append(round(validation_accuracy,4))
    
    #SAVE THE BEST MODEL
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        torch.save(lstm.state_dict(), 'best_model.pth')
        print("New best model\n")

#FINAL EVALUATION AND CLASSIFICATION REPORT
print("Training complete!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=['Fake', 'Real']))

all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        buffer_size = inputs.size(0)
        
        hidden_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
        cell_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
        
        outputs, _, _ = lstm(inputs, hidden_state, cell_state)
        outputs = outputs[:, -1, :]
        
        _, predicted = torch.max(outputs.data, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Test Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=['Fake', 'Real']))

#plots results
plot_training(NUM_EPOCHS, train_loss_list, val_loss_list, train_acc_list, val_acc_list)