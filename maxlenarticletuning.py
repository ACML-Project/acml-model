

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
# # 1) 1% loader for speed
# # ——————————————————————
# DATA_FRAC = 0.01
# _orig = Create_Datasets.Load_Merged_Data
# def small_loader():
#     return _orig().sample(frac=DATA_FRAC, random_state=42)
# Create_Datasets.Load_Merged_Data = small_loader
# Preprocessing.Load_Merged_Data = small_loader

# # ——————————————————————
# # 2) Your top‐3 length candidates
# # ——————————————————————
# candidates = [334, 491, 601]
# print(f"Evaluating MAX_ARTICLE_LEN candidates: {candidates}")

# # ——————————————————————
# # 3) Helper to run Train.py for exactly 1 epoch
# # ——————————————————————
# def run_once(max_len):
#     # rebuild pickles at this max_len
#     Preprocessing.MAX_ARTICLE_LEN = max_len
#     Preprocessing.Preprocess_Data()

#     # patch Train.py to load 1% data & run 1 epoch
#     with open("Train.py") as f:
#         orig = f.read()

#     header = f"""
# import os, sys
# sys.path.insert(0, os.getcwd())

# # force 1% loader
# import Create_Datasets
# _orig = Create_Datasets.Load_Merged_Data
# def small_cd(): return _orig().sample(frac={DATA_FRAC}, random_state=42)
# Create_Datasets.Load_Merged_Data = small_cd

# import Preprocessing
# Preprocessing.Load_Merged_Data = small_cd
# Preprocessing.Load_Data       = lambda: (
#     __import__('pickle').load(open(Preprocessing.PKL_PATH+'encoded_data.pkl','rb')),
#     __import__('pickle').load(open(Preprocessing.PKL_PATH+'vocab.pkl','rb'))
# )

# # only 1 epoch
# NUM_EPOCHS = 1
# """

#     tmp = tempfile.mktemp(prefix="tune_", suffix=".py")
#     with open(tmp, "w") as f:
#         f.write(header + orig)

#     proc = subprocess.run([sys.executable, tmp],
#                           capture_output=True, text=True)
#     os.remove(tmp)

#     # parse “Best Validation Accuracy”
#     for ln in reversed(proc.stdout.splitlines()):
#         m = re.match(r"Best Validation Accuracy:\s*([0-9.]+)%", ln)
#         if m:
#             return float(m.group(1))

#     raise RuntimeError("Validation accuracy not found")

# # ——————————————————————
# # 4) Loop over candidates
# # ——————————————————————
# best_len, best_acc = None, -1.0
# for L in candidates:
#     print(f"\n→ Testing max_len = {L}")
#     acc = run_once(L)
#     print(f"   Validation Accuracy = {acc:.2f}%")
#     if acc > best_acc:
#         best_len, best_acc = L, acc

# # ——————————————————————
# # 5) Report your winner
# # ——————————————————————
# print(f"\n Best MAX_ARTICLE_LEN = {best_len} → {best_acc:.2f}%")



import nltk
import random
import tempfile
import subprocess
import re
import os
import sys
import pandas as pd

import Preprocessing
import Create_Datasets


DATA_FRAC = 0.01
_orig = Create_Datasets.Load_Merged_Data

def small_loader():
    return _orig().sample(frac=DATA_FRAC, random_state=42)

Create_Datasets.Load_Merged_Data = small_loader
Preprocessing.Load_Merged_Data = small_loader

TRIALS, EPOCHS = 5, 5

# ——————————————————————
# Compute 75th & 95th percentiles on 10% subsample
# ——————————————————————
df = Create_Datasets.Load_Merged_Data()
lengths = [len(Preprocessing.Preprocess_Text(t)) for t in df['text']]
lo, hi = map(int, pd.Series(lengths).quantile([0.75, 0.95]))
print(f"Tuning MAX_ARTICLE_LEN between {lo} and {hi} ({TRIALS} trials, {EPOCHS} epoch each)")

best = (None, -1.0)

for i in range(1, TRIALS + 1):
    L = random.randint(lo, hi)
    print(f"\nTrial {i}/{TRIALS}: MAX_ARTICLE_LEN = {L}")

   
    Preprocessing.MAX_ARTICLE_LEN = L
    Preprocessing.Preprocess_Data()

  
    with open("Train.py", "r") as f:
        orig = f.read()

    header = f"""
import os, sys
sys.path.insert(0, os.getcwd())

# patch Create_Datasets to 1% loader
import Create_Datasets
_orig_cd = Create_Datasets.Load_Merged_Data
def small_cd(): return _orig_cd().sample(frac={DATA_FRAC}, random_state=42)
Create_Datasets.Load_Merged_Data = small_cd

# patch Preprocessing loaders to match subset
import Preprocessing
Preprocessing.Load_Merged_Data = small_cd
Preprocessing.Load_Data = lambda: (
    __import__('pickle').load(open(Preprocessing.PKL_PATH + 'encoded_data.pkl','rb')),
    __import__('pickle').load(open(Preprocessing.PKL_PATH + 'vocab.pkl','rb'))
)

# override epochs
NUM_EPOCHS = {EPOCHS}
"""

    tmp = tempfile.mktemp(prefix="tune_", suffix=".py")
    with open(tmp, "w") as f:
        f.write(header + orig)

    proc = subprocess.run(
        [sys.executable, tmp],
        capture_output=True,
        text=True
    )
    os.remove(tmp)


    acc = None
    for ln in proc.stdout.splitlines():
        m = re.match(r"Best Validation Accuracy:\s*([0-9.]+)%", ln)
        if m:
            acc = float(m.group(1))
            break

    if acc is None:
        print("Could not find accuracy in Train.py output.")
        print("=== stdout ===\n", proc.stdout or "(no stdout)")
        print("=== stderr ===\n", proc.stderr or "(no stderr)")
        continue

    print(f" → Validation Accuracy = {acc:.2f}%")
    if acc > best[1]:
        best = (L, acc)


if best[0] is not None:
    print(f"\n Best MAX_ARTICLE_LEN = {best[0]} → {best[1]:.2f}%")
else:
    print("\nNo valid trials completed—check Train.py output above")
