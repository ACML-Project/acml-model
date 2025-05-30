from sklearn.manifold import TSNE
from Create_Datasets import Load_Merged_Data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from Preprocessing import Save_Get_Untruncated
import random
import re
import subprocess
import sys
import tempfile


DATA_FRAC = 0.01
FIGSIZE_1 = (6, 6)
FIGSIZE_2 = (12, 6)
TRIALS = 5
EPOCHS = 5


#FOR CREATING BASIC BAR GRAPHS
def Create_Bar_Graph(x, y, labels, figsize):

    width = 0.6

    plt.figure(figsize=figsize)
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])

    bars = plt.bar(x, y, width=width, align='center')

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + width/2,
            height,
            f'{height}',
            ha='center',
            va='bottom'
        )

    plt.show()


#CHECKING IF THERE'S AN IMBALANCE BETWEEN FAKE AND REAL NEWS ARTICLES
def Label_Frequency_Graph(dataset):

    labels = ['Fake', 'Real']
    counter = [0, 0]

    #[TITLE, X, Y]
    graph_labels = [
        'Number of fake vs real news articles',
        'Article classification', 
        'Number of articles' 
    ]

    for i in dataset['label']:
        counter[i] += 1

    Create_Bar_Graph(labels, counter, graph_labels, FIGSIZE_1)
    return


def Subject_Frequency_Graph(dataset):

    subjects = ['politics', 'worldnews', 'Government News', 'News', 'politicsNews', 'left-news', 'Middle-east', 'US_News']
    x_axis = ['Politics', 'World News', 'Government News', 'News', 'Politics News', 'Left News', 'Middle-east', 'US News']
    counter = [0]*len(subjects)

    #[TITLE, X, Y]
    graph_labels = [
        'Number of articles per subject matter',
        'Subject',
        'Number of articles'
    ]

    for subject in dataset['subject']:
        for i in range(len(subjects)):
            if subject == subjects[i]:
                counter[i] = counter[i] + 1
                break

    Create_Bar_Graph(x_axis, counter, graph_labels, FIGSIZE_2)
    return


def Subject_Label_Graph(dataset):

    x_axis = ['Fake', 'Real'] 
    subjects = ['politics', 'worldnews', 'Government News', 'News', 'politicsNews', 'left-news', 'Middle-east', 'US_News']
    subject_labels = ['Politics', 'World News', 'Government News', 'News', 'Politics News', 'Left News', 'Middle-east', 'US News']
    n = len(subjects)

    #[FAKE, REAL]
    subject_counter = [[0]*n, [0]*n]
    label_counter = [0, 0]

    #[TITLE, X, Y]
    graph_labels = [
        'Number of articles by article classification and subject matter',
        'Article classification',
        'Number of articles'
    ]

    for i in range(len(dataset)):
        for j in range(n):
            if dataset['subject'][i] == subjects[j]:
                label = dataset['label'][i]
                subject_counter[label][j] = subject_counter[label][j] + 1
                break

    num_bars = 2
    for i in range(num_bars): label_counter[i] = sum(subject_counter[i])
    subject_counter = np.transpose(subject_counter)
    plt.figure(figsize=FIGSIZE_1)

    for i in range(n):
        if i != 0:

            counter_sum = [0, 0]
            for j in range(i):
                counter_sum += subject_counter[j]

            plt.bar(x_axis, subject_counter[i], bottom=counter_sum)
        else: plt.bar(x_axis, subject_counter[0])
    
    plt.title(graph_labels[0])
    plt.xlabel(graph_labels[1])
    plt.ylabel(graph_labels[2])
    plt.legend(subject_labels, loc='upper left', bbox_to_anchor=(1,1))
    plt.show()
    return


def Plot_tsne(test_embeddings, test_labels, save_path='Cooler Graphs/tsne.png'):

    embeddings_array = np.array(test_embeddings)
    labels_array = np.array(test_labels)

    #reduce to 2D with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings_array)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels_array, cmap='coolwarm', alpha=0.6)
    plt.colorbar(label='Class')
    plt.title('t-SNE of LSTM Embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


def Plot_Histo(test_probs, test_labels, save_path='Basic Graphs/histo.png'):

    plt.hist([p for p, t in zip(test_probs, test_labels) if t == 0], bins=30, alpha=0.5, label='Fake')
    plt.hist([p for p, t in zip(test_probs, test_labels) if t == 1], bins=30, alpha=0.5, label='Real')
    plt.legend()
    plt.title("Predicted Probabilities by Class")
    plt.savefig(save_path)
    plt.show()

    
def Plot_Confusion_Matrix(cm, title, save_path):
    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - ' + title)
    plt.savefig(save_path)
    plt.show()


def Plot_Training(NUM_EPOCHS, train_loss_list, val_loss_list, train_acc_list, val_acc_list, save_path='Basic Graphs/training.png'):
    
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


def Remove_Outliers(lengths):

    q1 = np.percentile(lengths, 25)
    q3 = np.percentile(lengths, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr

    return [ l for l in lengths if lower_bound <= l <= upper_bound ]


def Violin_BoxPlot_Article_Length(lengths, save_path):

    lengths = Remove_Outliers(lengths)

    plt.violinplot(
        dataset = lengths,
        vert = False,
        showextrema = False
    )

    plt.boxplot(lengths, vert=False, showfliers=False)
    plt.title("A violin plot and a boxplot of the length of preprocessed articles")
    plt.xlabel("Length of articles")
    plt.ylabel("Number of articles")
    plt.gca().yaxis.set_visible(False)
    plt.savefig(save_path)
    plt.show()


dataset = Load_Merged_Data()
untruncated_dataset = Save_Get_Untruncated(False)
article_lengths = [ len(article) for article in untruncated_dataset ]

#BASIC ASS GRAPHS
#Label_Frequency_Graph(dataset)
#Subject_Frequency_Graph(dataset)

#COOLER GRAPH
#Subject_Label_Graph(dataset)
#Violin_BoxPlot_Article_Length(article_lengths, 'Basic Graphs/Article length.png')