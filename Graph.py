import matplotlib.pyplot as plt
def plot_training(NUM_EPOCHS, train_loss_list, val_loss_list, train_acc_list, val_acc_list, save_path='graph.png'):
    
    #listing epochs
    epochs = list(range(1, NUM_EPOCHS+1))
    
    #plotting loss between training and validation data
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, val_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    #plotting accuracy between training and validation data
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, label='Train Acc')
    plt.plot(epochs, val_acc_list, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
