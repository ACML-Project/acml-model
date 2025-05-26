from Create_Datasets import Split_Dataset, Load_Merged_Data
from LSTM import LSTM
from Preprocessing import Load_Data, Create_Readable_Text
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report


# IMPROVED HYPERPARAMETERS
BATCH_SIZE = 64  # Increased for more stable gradients
DROP_OUT = 0.2  # Enable dropout for regularization
EMBEDDING_DIM = 256  # Increased for richer representations
HIDDEN_SIZE = 256  # Increased for more complex patterns
INITIAL_LEARNING_RATE = 0.001  # Lowered for more stable training
NUM_EPOCHS = 25  # Increased with early stopping
NUM_RECURRENT_LAYERS = 2  # Stacked LSTM for deeper learning
WEIGHT_DECAY = 1e-4  # L2 regularization
MAX_GRAD_NORM = 1.0  # Gradient clipping
PATIENCE = 5  # Early stopping patience
MIN_DELTA = 0.001  # Minimum improvement threshold


# GET DATA FROM PICKLE JAR
unprocessed_data = Load_Merged_Data() 
encoded_data, vocab = Load_Data()

# IF YOU WANT TO READ DATA
# Create_Readable_Text(unprocessed=unprocessed_data, encoding=encoded_data, vocab=vocab)

# CREATES TENSORS FOR THE BINARY LABELS (FAKE/REAL NEWS) AND FOR THE ENCODED TEXT
labels = torch.tensor(unprocessed_data['label'].values, dtype=torch.long)
inputs = torch.tensor(encoded_data, dtype=torch.long)

dataset = TensorDataset(inputs, labels)
training_data, validation_data, test_data = Split_Dataset(dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# DATALOADERS MAKE IT EASIER TO LOAD AND PROCESS LARGE DATASETS IN PARALLEL
training_loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(dataset=validation_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

# CREATE LSTM MODEL WITH IMPROVED PARAMETERS
lstm = LSTM(
    dropout=DROP_OUT,  # Enable dropout
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_recurrent_layers=NUM_RECURRENT_LAYERS,
    output_size=2,  # BECAUSE IT'S A BINARY CLASSIFICATION
    len_vocab=len(vocab),
).to(device)

# COMPILE MODEL FOR FASTER COMPUTATION
nn.Module.compile(lstm)

# LOSS FUNCTION AND OPTIMIZER WITH IMPROVEMENTS
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), 
                           lr=INITIAL_LEARNING_RATE, 
                           weight_decay=WEIGHT_DECAY)

# LEARNING RATE SCHEDULER
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='max', 
                                                     patience=3, 
                                                     factor=0.5)

# EARLY STOPPING AND BEST MODEL TRACKING
best_validation_accuracy = 0
early_stopping_counter = 0

print("Starting training with improved hyperparameters...")
print(f"Model parameters: Hidden={HIDDEN_SIZE}, Layers={NUM_RECURRENT_LAYERS}, Dropout={DROP_OUT}")
print(f"Training parameters: LR={INITIAL_LEARNING_RATE}, Batch={BATCH_SIZE}, Max Epochs={NUM_EPOCHS}")
print("-" * 70)

# TRAINING AND VALIDATION LOOP
for epoch in range(NUM_EPOCHS):
    
    # TRAINING PHASE
    lstm.train()
    training_loss = 0
    training_correct = 0
    training_total = 0
    
    for inputs, labels in training_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        buffer_size = inputs.size(0)
        
        hidden_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
        cell_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
        
        # FORWARD PASS
        optimizer.zero_grad()
        outputs, hidden, memory = lstm(inputs, hidden_state, cell_state)
        outputs = outputs[:, -1, :]  # TAKE THE LAST OUTPUT FOR CLASSIFICATION
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # GRADIENT CLIPPING TO PREVENT EXPLODING GRADIENTS
        torch.nn.utils.clip_grad_norm_(lstm.parameters(), MAX_GRAD_NORM)
        
        optimizer.step()
        
        # METRIC CALCULATIONS
        training_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        training_total += labels.size(0)
        training_correct += (predicted == labels).sum().item()
    
    # VALIDATION PHASE
    lstm.eval()
    validation_loss = 0
    validation_correct = 0
    validation_total = 0
    
    # FOR CLASSIFICATION REPORT
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            buffer_size = inputs.size(0)
            
            hidden_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
            cell_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
            
            outputs, hidden, memory = lstm(inputs, hidden_state, cell_state)
            outputs = outputs[:, -1, :]  # TAKE THE LAST OUTPUT FOR CLASSIFICATION
            
            loss = loss_fn(outputs, labels)
            validation_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            validation_total += labels.size(0)
            validation_correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # CALCULATE METRICS FOR EPOCH
    training_loss = training_loss / len(training_loader)
    training_accuracy = 100 * training_correct / training_total
    validation_loss = validation_loss / len(validation_loader)
    validation_accuracy = 100 * validation_correct / validation_total
    
    # UPDATE LEARNING RATE BASED ON VALIDATION PERFORMANCE
    scheduler.step(validation_accuracy)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}:')
    print(f'Train Loss: {training_loss:.4f} | Acc: {training_accuracy:.2f}%')
    print(f'Val Loss: {validation_loss:.4f} | Acc: {validation_accuracy:.2f}%')
    print(f'Learning Rate: {current_lr:.6f}')
    
    # EARLY STOPPING AND BEST MODEL SAVING
    if validation_accuracy > best_validation_accuracy + MIN_DELTA:
        best_validation_accuracy = validation_accuracy
        early_stopping_counter = 0
        torch.save(lstm.state_dict(), 'best_model.pth')
        print("✓ New best model saved!")
    else:
        early_stopping_counter += 1
        print(f"No improvement ({early_stopping_counter}/{PATIENCE})")
        
        if early_stopping_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation accuracy: {best_validation_accuracy:.2f}%")
            break
    
    print("-" * 50)

# FINAL EVALUATION AND CLASSIFICATION REPORT
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

# OPTIONAL: EVALUATE ON TEST SET
print("\n" + "-"*50)
print("EVALUATING ON TEST SET:")
print("-"*50)

lstm.eval()
test_predictions = []
test_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        buffer_size = inputs.size(0)
        
        hidden_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
        cell_state = torch.zeros(NUM_RECURRENT_LAYERS, buffer_size, HIDDEN_SIZE).to(device)
        
        outputs, hidden, memory = lstm(inputs, hidden_state, cell_state)
        outputs = outputs[:, -1, :]
        
        _, predicted = torch.max(outputs.data, 1)
        test_predictions.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("\nTest Set Classification Report:")
print(classification_report(test_labels, test_predictions, target_names=['Fake', 'Real']))