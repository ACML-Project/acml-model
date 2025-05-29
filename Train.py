from Create_Datasets import Load_Merged_Data
from Graphs import Plot_Training, Plot_Confusion_Matrix, Plot_Histo, Plot_tsne
from LSTM import LSTM
from Preprocessing import Load_Data, Create_Readable_Text
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


#HYPERPARMETER TUNING
BATCH_SIZE = 16
DROP_OUT = 0.5 #ONLY NECESSARY IF USING A STACKED LSTM
EMBEDDING_DIM = 128 #SIZE OF THE VECTOR FOR EACH EMBEDDING
HIDDEN_SIZE = 128 #NUMBER OF FEATURES FOR THE HIDDEN STATE
NUM_EPOCHS = 20
NUM_RECURRENT_LAYERS = 2 #CREATES A STACKED LSTM IF >1.
MAX_GRAD_NORM = 1.0 # Gradient clipping
PATIENCE = 5  # Earlier stopping to save time
WEIGHT_DECAY = 1e-5 #L2 REGULARIZATION
learning_rate = 0.001 #NO LONGER A CONSTANT, WILL CHANGE AS MODEL TRAINS


def Create_Dataloaders(datasets, labels):

    tensor_datasets = []

    for i in range(3):

        #CREATES TENSORS FOR THE BINARY LABELS (FAKE/REAL NEWS) AND FOR THE ENCODED TEXT
        label = torch.tensor(labels[i], dtype=torch.long)
        inputs = torch.tensor(datasets[i], dtype=torch.long)

        tensor_dataset = TensorDataset(inputs, label)
        tensor_datasets.append(tensor_dataset)

    #DATALOADERS MAKE IT EASIER TO LOAD AND PROCESS LARGE DATASETS IN PARALLEL
    training_loader = DataLoader(dataset=tensor_datasets[0], batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(dataset=tensor_datasets[1], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=tensor_datasets[2], batch_size=BATCH_SIZE, shuffle=True)
    
    return training_loader, validation_loader, test_loader


#GET DATA FROM PICKLE JAR 
encoded_data, labels, vocab = Load_Data()
 
#IF YOU WANT TO READ DATA
unprocessed_data = Load_Merged_Data()
Create_Readable_Text(unprocessed_data, encoded_data, labels, vocab)

device = torch.device(0 if torch.cuda.is_available() else 'cpu')
training_loader, validation_loader, test_loader = Create_Dataloaders(encoded_data, labels)

lstm = LSTM(
    dropout = DROP_OUT, #CHANGE IN LSTM.PY
    embedding_dim = EMBEDDING_DIM,
    hidden_size = HIDDEN_SIZE,
    num_recurrent_layers = NUM_RECURRENT_LAYERS,
    output_size = 2, #BECAUSE IT'S A BINARY CLASSIFICATION
    len_vocab = len(vocab),
).to(device)

nn.Module.compile(lstm) #SHOULD MAKE COMPUTATION FASTER
nn.utils.clip_grad_norm_(lstm.parameters(), max_norm=MAX_GRAD_NORM)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode ='max', 
    patience = PATIENCE, 
    factor=0.5
)

early_stopping_counter = 0
best_validation_accuracy = 0

#for graphing
train_loss_list = []
val_loss_list = []
train_acc_list =  []
val_acc_list =    []


print("Starting training with improved hyperparameters...")
print(f"Model parameters: Hidden = {HIDDEN_SIZE}, Layers = {NUM_RECURRENT_LAYERS}, Dropout = {DROP_OUT}")
print(f"Training parameters: LR = {learning_rate}, Batch = {BATCH_SIZE}, Max Epochs = {NUM_EPOCHS}")
print("-" * 70)

#TRAINING AND VALIDATION
for epoch in range(NUM_EPOCHS):

    lstm.train()
    training_loss = 0
    training_correct = 0
    training_total = 0
    
    for inputs, labels in training_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        buffer_size = inputs.size(0)
        
        hidden_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
        cell_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
        
        #FOWARD PASS
        optimizer.zero_grad()
        outputs, hidden, memory = lstm(inputs, hidden_state, cell_state)
        outputs = outputs[:, -1, :] #TAKE THE LAST OUTPUT FOR CLASSIFICATION
        
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

    # UPDATE LEARNING RATE BASED ON VALIDATION PERFORMANCE
    scheduler.step(validation_accuracy)
    learning_rate = optimizer.param_groups[0]['lr']

    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}:')
    print(f'Train Loss: {training_loss:.4f} | Acc: {training_accuracy:.2f}%')
    print(f'Val Loss: {validation_loss:.4f} | Acc: {validation_accuracy:.2f}%\n')

    #saving results for graphing
    train_loss_list.append(round(training_loss,4))
    val_loss_list.append(round(validation_loss,4))
    train_acc_list.append(round(training_accuracy,4))
    val_acc_list.append(round(validation_accuracy,4))
    
    #SAVE THE BEST MODEL
    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy
        early_stopping_counter = 0
        torch.save(lstm.state_dict(), 'best_model.pth')
        print("✓ New best model saved! \n")

    else:
        early_stopping_counter += 1
        print(f"No improvement ({early_stopping_counter}/{PATIENCE})")
        
        if early_stopping_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation accuracy: {best_validation_accuracy:.2f}%")
            break
    
    print("-" * 50)


print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"Best Validation Accuracy: {best_validation_accuracy:.2f}%")
print(f"Final Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

# LOAD BEST MODEL FOR FINAL EVALUATION
lstm.load_state_dict(torch.load('best_model.pth'))
print("\nLoaded best model for final evaluation")

print("\nFinal Classification Report (Validation Set):")
print(classification_report(all_labels, all_predictions, target_names=['Fake', 'Real']))

#EVALUATE ON TEST SET
print("\n" + "-"*50)
print("EVALUATING ON TEST SET:")
print("-"*50)

lstm.eval()
test_predictions = []
test_labels = []    
test_probs = [] #collects the probability of the positive class for each sample.
test_embeddings = [] 

with torch.no_grad():
    for inputs, labels in test_loader:
    
        inputs, labels = inputs.to(device), labels.to(device)
        buffer_size = inputs.size(0)
        
        hidden_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
        cell_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
        
        outputs, _, _ = lstm(inputs, hidden_state, cell_state)
        outputs = outputs[:, -1, :]
        
        probs = torch.softmax(outputs, dim=1)
        test_probs.extend(probs[:, 1].cpu().numpy()) 
        
        _, predicted = torch.max(outputs.data, 1)
        
        test_predictions.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

        test_embeddings.extend(outputs.cpu().numpy()) 

test_accuracy = 100 * accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("\nTest Set Classification Report:")
print(classification_report(test_labels, test_predictions, target_names=['Fake', 'Real']))
print("\nTest Set Confusion Matrix:")
cm = confusion_matrix(test_labels, test_predictions)
print(cm)
Plot_Confusion_Matrix(cm)
Plot_Histo(test_probs, test_labels)
Plot_tsne(test_embeddings, test_labels)
#plots results
Plot_Training(len(train_loss_list), train_loss_list, val_loss_list, train_acc_list, val_acc_list)