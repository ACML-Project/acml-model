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

    
def Plot_Confusion_Matrix(cm, save_path='Basic Graphs/confusion.png'):
    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Test Dataset')
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


dataset = Load_Merged_Data()
untruncated_dataset = Save_Get_Untruncated(False)

#BASIC ASS GRAPHS
#Label_Frequency_Graph(dataset)
#Subject_Frequency_Graph(dataset)

#COOLER GRAPH
#Subject_Label_Graph(dataset)


# ——————————————————————
# Compute 75th & 95th percentiles
# ——————————————————————
lengths = [len(article) for article in untruncated_dataset]
lo, hi = map(int, pd.Series(lengths).quantile([0.9, 0.95]))
print(f"Tuning MAX_ARTICLE_LEN between {lo} and {hi} ({TRIALS} trials, {EPOCHS} epoch each)")

best = (None, -1.0)

#for i in range(1, TRIALS + 1):
 #   L = random.randint(lo, hi)
  #  print(f"\nTrial {i}/{TRIALS}: MAX_ARTICLE_LEN = {L}")

   
#    Preprocessing.MAX_ARTICLE_LEN = L
 #   Preprocessing.Preprocess_Data()

  
  #  with open("Train.py", "r") as f:
   #     orig = f.read()

    #header = f"""
#import os, sys
#sys.path.insert(0, os.getcwd())

# patch Create_Datasets to 1% loader
#import Create_Datasets
#_orig_cd = Create_Datasets.Load_Merged_Data
#def small_cd(): return _orig_cd().sample(frac={DATA_FRAC}, random_state=42)
#Create_Datasets.Load_Merged_Data = small_cd

# patch Preprocessing loaders to match subset
#import Preprocessing
#Preprocessing.Load_Merged_Data = small_cd
#Preprocessing.Load_Data = lambda: (
 #   __import__('pickle').load(open(Preprocessing.PKL_PATH + 'encoded_data.pkl','rb')),
  #  __import__('pickle').load(open(Preprocessing.PKL_PATH + 'vocab.pkl','rb'))
#)

# override epochs
#NUM_EPOCHS = {EPOCHS}
#"""

 #   tmp = tempfile.mktemp(prefix="tune_", suffix=".py")
  #  with open(tmp, "w") as f:
   #     f.write(header + orig)

    #proc = subprocess.run(
     #   [sys.executable, tmp],
      #  capture_output=True,
       # text=True
    #)
    #os.remove(tmp)


    #acc = None
    #for ln in proc.stdout.splitlines():
     #   m = re.match(r"Best Validation Accuracy:\s*([0-9.]+)%", ln)
      #  if m:
       #     acc = float(m.group(1))
        #    break

    #if acc is None:
     #   print("Could not find accuracy in Train.py output.")
      #  print("=== stdout ===\n", proc.stdout or "(no stdout)")
       # print("=== stderr ===\n", proc.stderr or "(no stderr)")
        #continue

  #  print(f" → Validation Accuracy = {acc:.2f}%")
   # if acc > best[1]:
    #    best = (L, acc)


#if best[0] is not None:
 #   print(f"\n Best MAX_ARTICLE_LEN = {best[0]} → {best[1]:.2f}%")
#else:
 #   print("\nNo valid trials completed—check Train.py output above")

    

# import os, random, subprocess, sys, re, tempfile
# import pandas as pd

# import Preprocessing
# import Create_Datasets

# # -------------------------------
# # Speed-ups: use only 1% of data
# # -------------------------------
# DATA_FRACTION = 0.01
# _orig_loader = Create_Datasets.Load_Merged_Data
# def small_loader():
#     df = _orig_loader()
#     return df.sample(frac=DATA_FRACTION, random_state=42)
# Create_Datasets.Load_Merged_Data = small_loader
# Preprocessing.Load_Merged_Data  = small_loader

# # -------------------------------
# # Random-search settings
# # -------------------------------
# N_TRIALS    = 5    # number of random lengths to try
# TRIAL_EPOCHS = 1   # epochs per trial

# # -----------------------------------------------
# # Step 1: determine your percentile bounds once
# # -----------------------------------------------
# def get_percentile_bounds():
#     # temporarily disable truncation
#     old = Preprocessing.MAX_ARTICLE_LEN
#     Preprocessing.MAX_ARTICLE_LEN = 10_000_000

#     df = Create_Datasets.Load_Merged_Data()
#     lengths = [len(Preprocessing.Preprocess_Text(txt)) for txt in df['text']]

#     Preprocessing.MAX_ARTICLE_LEN = old
#     pct = pd.Series(lengths).quantile([0.75, 0.95])
#     return int(pct.loc[0.75]), int(pct.loc[0.95])

# # -----------------------------------------------
# # Step 2: run a patched Train.py with fewer epochs
# # -----------------------------------------------
# def run_patched_train():
#     # Read original Train.py
#     with open('Train.py','r') as f:
#         orig = f.read()

#     # Build monkey‐patch + epoch override + path fix
#     patched = """
# import os, sys
# # ensure project folder is on path
# sys.path.insert(0, os.getcwd())

# # --- patch data loading to 1% ---
# import Create_Datasets
# _orig = Create_Datasets.Load_Merged_Data
# def small_loader():
#     df = _orig()
#     return df.sample(frac=0.01, random_state=42)
# Create_Datasets.Load_Merged_Data = small_loader

# # override epochs
# NUM_EPOCHS = {epochs}

# """.format(epochs=TRIAL_EPOCHS) + orig

#     # Write to temp file
#     fd, tmp = tempfile.mkstemp(prefix='Train_tune_', suffix='.py')
#     os.close(fd)
#     with open(tmp,'w') as f:
#         f.write(patched)

#     # Run it
#     proc = subprocess.run([sys.executable, tmp], capture_output=True, text=True)
#     os.remove(tmp)

#     # Parse accuracy
#     for line in reversed(proc.stdout.splitlines()):
#         m = re.search(r'Best Validation Accuracy: *([\d\.]+)%', line)
#         if m:
#             return float(m.group(1))

#     print("=== stdout ===\n", proc.stdout)
#     print("=== stderr ===\n", proc.stderr)
#     raise RuntimeError("Couldn't parse accuracy")
# # -----------------------------------------------
# # Main: random-search over lengths
# # -----------------------------------------------
# def random_search():
#     lo, hi = get_percentile_bounds()
#     print(f"Searching random lengths between {lo} and {hi}")

#     best = (None, -1.0)
#     for i in range(1, N_TRIALS+1):
#         L = random.randint(lo, hi)
#         print(f"\nTrial {i}/{N_TRIALS}: MAX_ARTICLE_LEN = {L}")

#         Preprocessing.MAX_ARTICLE_LEN = L
#         Preprocessing.Preprocess_Data()         # rebuild pickles on 1% data

#         acc = run_patched_train()
#         print(f" → Validation Accuracy = {acc:.2f}%")

#         if acc > best[1]:
#             best = (L, acc)

#     print(f"\n Best MAX_ARTICLE_LEN = {best[0]} → {best[1]:.2f}%")

# if __name__ == "__main__":
#     random_search()
# hyperparameters.py

# import os
# import random
# import subprocess
# import sys
# import re
# import tempfile

# import pandas as pd

# import Preprocessing
# import Create_Datasets

# # —————————————————————————————————————
# # 1% subsample of the dataset for speed
# # —————————————————————————————————————
# DATA_FRAC    = 0.10
# _orig_loader = Create_Datasets.Load_Merged_Data

# def small_loader():
#     return _orig_loader().sample(frac=DATA_FRAC, random_state=42)

# # Patch both global loaders so preprocessing & training see the same 1%
# Create_Datasets.Load_Merged_Data = small_loader
# Preprocessing.Load_Merged_Data    = small_loader

# # —————————————————————————————————————
# # Tuning settings
# # —————————————————————————————————————
# TRIALS      = 5                      # number of random trials
# EPOCHS      = 1                      # epochs per trial (speed)
# P_LOW, P_HIGH = 0.75, 0.95           # percentile bounds for max_len

# BATCH_CHOICES = [16, 32, 64]
# LR_CHOICES    = [1e-2, 5e-3, 1e-3]

# # —————————————————————————————————————
# # 1) Compute 75th & 95th percentiles on 1% data
# # —————————————————————————————————————
# df      = Create_Datasets.Load_Merged_Data()
# lengths = [len(Preprocessing.Preprocess_Text(t)) for t in df['text']]
# lo, hi  = map(int, pd.Series(lengths).quantile([P_LOW, P_HIGH]))
# print(f"Tuning over max_len ∈ [{lo}, {hi}], batch ∈ {BATCH_CHOICES}, lr ∈ {LR_CHOICES}")

# best_cfg = None
# best_acc = -1.0

# for i in range(1, TRIALS + 1):
#     # 2) Sample a random configuration
#     max_len    = random.randint(lo, hi)
#     batch_size = random.choice(BATCH_CHOICES)
#     lr         = random.choice(LR_CHOICES)
#     print(f"\nTrial {i}/{TRIALS}: max_len={max_len}, batch_size={batch_size}, lr={lr}")

#     # 3) Rebuild data at this max_len
#     Preprocessing.MAX_ARTICLE_LEN = max_len
#     Preprocessing.Preprocess_Data()

#     # 4) Create a patched Train.py that sets epochs, batch_size, lr, and 1% loader
#     with open("Train.py", "r") as f:
#         orig = f.read()

#     header = f"""
# import os, sys
# # ensure project folder on path
# sys.path.insert(0, os.getcwd())

# # patch data loader to 1%
# import Create_Datasets
# _orig = Create_Datasets.Load_Merged_Data
# def small_loader():
#     return _orig().sample(frac={DATA_FRAC}, random_state=42)
# Create_Datasets.Load_Merged_Data = small_loader

# # override hyperparameters
# NUM_EPOCHS     = {EPOCHS}
# BATCH_SIZE     = {batch_size}
# LEARNING_RATE  = {lr}

# """

#     patched = header + orig

#     # 5) Write patched script to temp file
#     fd, tmp_path = tempfile.mkstemp(prefix="Train_tune_", suffix=".py")
#     os.close(fd)
#     with open(tmp_path, "w") as f:
#         f.write(patched)

#     # 6) Run the patched training script
#     proc = subprocess.run([sys.executable, tmp_path],
#                           capture_output=True, text=True)
#     os.remove(tmp_path)

#     # 7) Parse the “Best Validation Accuracy: XX.XX%” line
#     acc = None
#     for line in reversed(proc.stdout.splitlines()):
#         m = re.search(r"Best Validation Accuracy: *([\d\.]+)%", line)
#         if m:
#             acc = float(m.group(1))
#             break

#     if acc is None:
#         print("=== stdout ===\n", proc.stdout)
#         print("=== stderr ===\n", proc.stderr)
#         raise RuntimeError("Failed to parse accuracy")

#     print(f" → Validation Accuracy = {acc:.2f}%")
#     if acc > best_acc:
#         best_acc = acc
#         best_cfg = (max_len, batch_size, lr)

# # 8) Report best configuration
# print(f"\n Best config: max_len={best_cfg[0]}, batch_size={best_cfg[1]}, lr={best_cfg[2]} → {best_acc:.2f}%")


# hyperparameters.py
#73
# import random
# import tempfile
# import subprocess
# import re
# import os
# import sys
# import pandas as pd

# import Preprocessing
# import Create_Datasets

# # ——————————————————————
# # Speed-up: 1% loader globally
# # ——————————————————————
# DATA_FRAC = 0.01
# _orig = Create_Datasets.Load_Merged_Data

# def small_loader():
#     return _orig().sample(frac=DATA_FRAC, random_state=42)

# Create_Datasets.Load_Merged_Data = small_loader
# Preprocessing.Load_Merged_Data = small_loader

# # ——————————————————————
# # Tuning settings
# # ——————————————————————
# TRIALS, EPOCHS = 5, 1

# # ——————————————————————
# # Compute 75th & 95th percentiles on 1% subsample
# # ——————————————————————
# df = Create_Datasets.Load_Merged_Data()
# lengths = [len(Preprocessing.Preprocess_Text(t)) for t in df['text']]
# lo, hi = map(int, pd.Series(lengths).quantile([0.75, 0.95]))
# print(f"Tuning MAX_ARTICLE_LEN between {lo} and {hi} ({TRIALS} trials, {EPOCHS} epoch each)")

# best = (None, -1.0)

# for i in range(1, TRIALS + 1):
#     L = random.randint(lo, hi)
#     print(f"\nTrial {i}/{TRIALS}: MAX_ARTICLE_LEN = {L}")

#     # 1) Rebuild pickles at this length
#     Preprocessing.MAX_ARTICLE_LEN = L
#     Preprocessing.Preprocess_Data()

#     # 2) Patch and run Train.py
#     with open("Train.py", "r") as f:
#         orig = f.read()

#     header = f"""
# import os, sys
# sys.path.insert(0, os.getcwd())

# # patch Create_Datasets to 1% loader
# import Create_Datasets
# _orig_cd = Create_Datasets.Load_Merged_Data
# def small_cd(): return _orig_cd().sample(frac={DATA_FRAC}, random_state=42)
# Create_Datasets.Load_Merged_Data = small_cd

# # patch Preprocessing loaders to match subset
# import Preprocessing
# Preprocessing.Load_Merged_Data = small_cd
# Preprocessing.Load_Data = lambda: (
#     __import__('pickle').load(open(Preprocessing.PKL_PATH + 'encoded_data.pkl','rb')),
#     __import__('pickle').load(open(Preprocessing.PKL_PATH + 'vocab.pkl','rb'))
# )

# # override epochs
# NUM_EPOCHS = {EPOCHS}
# """

#     tmp = tempfile.mktemp(prefix="tune_", suffix=".py")
#     with open(tmp, "w") as f:
#         f.write(header + orig)

#     proc = subprocess.run(
#         [sys.executable, tmp],
#         capture_output=True,
#         text=True
#     )
#     os.remove(tmp)

#     # 3) Parse accuracy
#     acc = None
#     for ln in proc.stdout.splitlines():
#         m = re.match(r"Best Validation Accuracy:\s*([0-9.]+)%", ln)
#         if m:
#             acc = float(m.group(1))
#             break

#     if acc is None:
#         print(" Could not find accuracy in Train.py output.")
#         print("=== stdout ===\n", proc.stdout or "(no stdout)")
#         print("=== stderr ===\n", proc.stderr or "(no stderr)")
#         continue

#     print(f" → Validation Accuracy = {acc:.2f}%")
#     if acc > best[1]:
#         best = (L, acc)

# # 4) Report
# if best[0] is not None:
#     print(f"\n Best MAX_ARTICLE_LEN = {best[0]} → {best[1]:.2f}%")
# else:
#     print("\nNo valid trials completed—check Train.py output above.")
# measure_lengths.py

# from Create_Datasets import Load_Merged_Data
# from Preprocessing import Preprocess_Text, MAX_ARTICLE_LEN as _OLD_MAX
# import pandas as pd

# # 1) Temporarily disable any truncation
# import Preprocessing
# Preprocessing.MAX_ARTICLE_LEN = 10_000_000

# # 2) Load all articles
# df = Load_Merged_Data()

# # 3) Tokenize each article and record its length
# lengths = []
# for i, text in enumerate(df['text'], 1):
#     tokens = Preprocess_Text(text)
#     lengths.append(len(tokens))
#     if i % 1000 == 0:
#         print(f"  Processed {i} / {len(df)} articles…")

# # 4) Compute and display percentiles
# series = pd.Series(lengths)
# for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
#     print(f"{int(q*100)}th percentile: {int(series.quantile(q))} tokens")

# print(f"Max length: {series.max()} tokens")

# # 5) Restore original setting
# Preprocessing.MAX_ARTICLE_LEN = _OLD_MAX
# evaluate_max_len.py
