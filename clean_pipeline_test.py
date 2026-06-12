import os, random, json, pickle, hashlib
os.environ["TFHUB_CACHE_DIR"] = os.path.abspath("./tfhub_modules")

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Config
DATASET_DIR = "aug-dataset1"
CLASSES = ["belly_pain", "burping", "cold_hot", "discomfort", "hungry", "tired"]
SR = 16000
DURATION = 7
SEED = 42

np.random.seed(SEED)
random.seed(SEED)

print("Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet loaded.")

CLASS_MAPPING = {
    'bp': 'belly_pain',
    'bu': 'burping',
    'ch': 'cold_hot',
    'dc': 'discomfort',
    'hu': 'hungry',
    'ti': 'tired'
}

def clean_audio(y):
    return librosa.util.normalize(y)

def extract_features_yamnet(y):
    scores, embeddings, spectrogram = yamnet_model(y)
    yamnet_emb = embeddings.numpy()
    yamnet_mean = np.mean(yamnet_emb, axis=0)
    yamnet_max = np.max(yamnet_emb, axis=0)
    return np.concatenate([yamnet_mean, yamnet_max])

def load_wav_fixed(path):
    try:
        # 1. Enforce 8000 Hz uniform sample rate to discard high-frequency bias!
        y_8k, sr_8k = librosa.load(str(path), sr=8000, duration=DURATION, mono=True)
        if len(y_8k) < 8000 * 0.5:
            return None
        
        # 2. Resample to 16000 Hz for YAMNet
        y = librosa.resample(y_8k, orig_sr=8000, target_sr=16000)
        y = clean_audio(y)
        
        target_len = SR * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]
        return y.astype(np.float32)
    except:
        return None

# Load and cleanse dataset
print("Loading and resolving dataset...")
base_dir = Path(DATASET_DIR)
all_files = []
waveform_hashes = {}

for cls in CLASSES:
    cls_dir = base_dir / cls
    files = list(cls_dir.glob("*.wav"))
    for f in files:
        try:
            y, sr = librosa.load(str(f), sr=8000, duration=3.0)
            y_rounded = np.round(y, 3)
            h = hashlib.md5(y_rounded.tobytes()).hexdigest()
            
            name_lower = f.name.lower()
            detected_label = None
            for code, mapped_cls in CLASS_MAPPING.items():
                if f"-{code}" in name_lower or f"_{code}" in name_lower or name_lower.startswith(f"{code}-") or f"({code})" in name_lower or (code == 'bp' and 'bp' in name_lower):
                    detected_label = mapped_cls
                    break
            if not detected_label:
                detected_label = cls
                
            all_files.append({'path': f, 'hash': h, 'resolved_label': detected_label})
        except:
            pass

# Deduplicate
resolved_dataset = {}
for item in all_files:
    h = item['hash']
    resolved_dataset[h] = item

clean_dataset = list(resolved_dataset.values())
print(f"Total unique waveforms: {len(clean_dataset)}")

clean_paths = [item['path'] for item in clean_dataset]
clean_labels = [item['resolved_label'] for item in clean_dataset]

# Stratified Split
le = LabelEncoder()
y_enc = le.fit_transform(clean_labels)
idx = np.arange(len(clean_paths))

idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y_enc, random_state=SEED)

print(f"Train samples: {len(idx_tr)}")
print(f"Test samples: {len(idx_te)}")

# Extract features
X_tr, y_tr = [], []
X_te, y_te = [], []

print("\nExtracting training features...")
for count, i in enumerate(idx_tr):
    path = clean_paths[i]
    lbl = y_enc[i]
    y = load_wav_fixed(path)
    if y is not None:
        feat = extract_features_yamnet(y)
        X_tr.append(feat)
        y_tr.append(lbl)
    if (count+1) % 100 == 0:
        print(f"  Processed {count+1}/{len(idx_tr)}")

print("\nExtracting test features...")
for count, i in enumerate(idx_te):
    path = clean_paths[i]
    lbl = y_enc[i]
    y = load_wav_fixed(path)
    if y is not None:
        feat = extract_features_yamnet(y)
        X_te.append(feat)
        y_te.append(lbl)
    if (count+1) % 50 == 0:
        print(f"  Processed {count+1}/{len(idx_te)}")

X_tr, y_tr = np.array(X_tr), np.array(y_tr)
X_te, y_te = np.array(X_te), np.array(y_te)

# Train a Random Forest on resolved, clean, unbiased features
clf = RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight='balanced')
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)

print("\n=== Clean Resolved Pipeline Random Forest Results ===")
print(f"Stratified Test Accuracy: {accuracy_score(y_te, y_pred)*100:.2f}%")
print(classification_report(y_te, y_pred, target_names=le.classes_))
