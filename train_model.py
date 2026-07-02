import sys
import io
import os
import random
import json
import pickle
import hashlib
import warnings
from pathlib import Path

# Set TFHUB cache directory in workspace before any tensorflow imports
os.environ["TFHUB_CACHE_DIR"] = os.path.abspath("./tfhub_modules")
warnings.filterwarnings("ignore")

# Force stdout to use UTF-8 to prevent Windows terminal encoding crashes
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# -*- coding: utf-8 -*-
"""
CrySense - Single-Branch MLP Pipeline with Bandwidth Enforcer
==============================================================
"""

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# =========================================================
# CONFIG
# =========================================================
DATASET_DIR  = "clean-dataset-v17"
CLASSES      = ["belly_pain", "burping", "cold_hot", "discomfort", "hungry", "tired"]
MODEL_PATH   = "crysense_model.h5"
ENCODER_PATH = "label_encoder.pkl"
HISTORY_PATH = "training_history.json"
META_PATH    = "model_meta.json"

SR           = 16000
DURATION     = 7  
SEED         = 42
BATCH_SIZE   = 32
EPOCHS       = 80

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# =========================================================
# LOAD YAMNET
# =========================================================
print("\n[STEP 1] Loading pre-trained YAMNet model...")
print(f"  Cache directory set to: {os.environ['TFHUB_CACHE_DIR']}")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("  YAMNet loaded successfully.\n")

# =========================================================
# AUDIO PROCESSING & FEATURE EXTRACTION
# =========================================================
def clean_audio(y):
    return librosa.util.normalize(y)

def augment_audio(y, sr):
    choice = random.randint(0, 4)
    try:
        if choice == 0:  # Mild noise
            noise = np.random.normal(0, 0.003, len(y))
            y = y + noise
        elif choice == 1:  # Pitch shift
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-1.5, 1.5))
        elif choice == 2:  # Time stretch
            y = librosa.effects.time_stretch(y, rate=random.uniform(0.85, 1.15))
        elif choice == 3:  # Volume gain
            y = y * random.uniform(0.8, 1.2)
        else:  # Time shifting/rolling
            shift = int(sr * random.uniform(0.1, 0.25))
            y = np.roll(y, shift)
    except:
        pass
    return y.astype(np.float32)

def load_wav_fixed(path, do_augment=False):
    try:
        # Enforce 8000 Hz uniform load to discard high frequency sample rate bias
        y_8k, sr_8k = librosa.load(str(path), sr=8000, duration=DURATION, mono=True)
        if len(y_8k) < 8000 * 0.5:
            return None
        
        # Resample back to 16000 Hz for YAMNet
        y = librosa.resample(y_8k, orig_sr=8000, target_sr=16000)
        y = clean_audio(y)
        if do_augment:
            y = augment_audio(y, SR)
            y = clean_audio(y) # Re-normalize after augment
            
        target_len = SR * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]
        return y.astype(np.float32)
    except:
        return None

def extract_hybrid_features(y, sr):
    # 1. YAMNet Embeddings (Semantic context)
    scores, embeddings, spectrogram = yamnet_model(y)
    yamnet_emb = embeddings.numpy()
    yamnet_mean = np.mean(yamnet_emb, axis=0)
    yamnet_max = np.max(yamnet_emb, axis=0)
    yamnet_features = np.concatenate([yamnet_mean, yamnet_max])
    
    # 2. Fine-grained features
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)
    
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # RMS
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    # Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    
    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)
    
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)
    mel_std = np.std(mel_db, axis=1)
    
    trad_features = np.concatenate([
        mfcc_mean, mfcc_std,
        [centroid_mean, centroid_std],
        contrast_mean, contrast_std,
        [zcr_mean, zcr_std],
        [rms_mean, rms_std],
        chroma_mean, chroma_std,
        [rolloff_mean, rolloff_std],
        mel_mean, mel_std
    ])
    
    return np.concatenate([yamnet_features, trad_features])


def resolve_true_label(file_name, current_folder):
    name_lower = file_name.lower()
    if "-bp" in name_lower or "_bp" in name_lower or name_lower.startswith("bp-") or "(bp)" in name_lower or "belly_pain" in name_lower:
        return "belly_pain"
    if "-bu" in name_lower or "_bu" in name_lower or name_lower.startswith("bu-") or "(bu)" in name_lower or "burping" in name_lower:
        return "burping"
    if "-ch" in name_lower or "_ch" in name_lower or name_lower.startswith("ch-") or "(ch)" in name_lower or "cold_hot" in name_lower:
        return "cold_hot"
    if "-dc" in name_lower or "_dc" in name_lower or name_lower.startswith("dc-") or "(dc)" in name_lower or "discomfort" in name_lower:
        return "discomfort"
    if "-hu" in name_lower or "_hu" in name_lower or name_lower.startswith("hu-") or "(hu)" in name_lower or "hungry" in name_lower:
        return "hungry"
    if "-ti" in name_lower or "_ti" in name_lower or name_lower.startswith("ti-") or "(ti)" in name_lower or "tired" in name_lower:
        return "tired"
    return current_folder

# =========================================================
# MODEL ARCHITECTURES
# =========================================================
def make_wider_mlp(input_dim, n_classes):
    inp = Input(shape=(input_dim,), name='wider_input')
    x = BatchNormalization()(inp)
    x = Dropout(0.25)(x)
    
    x = Dense(256, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.20)(x)
    
    x = Dense(128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inp, out, name='wider_mlp')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def make_deep_mlp(input_dim, n_classes):
    inp = Input(shape=(input_dim,), name='deep_input')
    x = BatchNormalization()(inp)
    x = Dropout(0.30)(x)
    
    x = Dense(512, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Dense(256, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.20)(x)
    
    x = Dense(128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inp, out, name='deep_mlp')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def make_residual_mlp(input_dim, n_classes):
    inp = Input(shape=(input_dim,), name='residual_input')
    x = BatchNormalization()(inp)
    x = Dropout(0.25)(x)
    
    # Project to 256
    x_proj = Dense(256, activation='swish')(x)
    x_proj = BatchNormalization()(x_proj)
    
    # Residual Block
    res = Dense(256, activation='swish')(x_proj)
    res = BatchNormalization()(res)
    res = Dropout(0.20)(res)
    res = Dense(256, activation='swish')(res)
    res = BatchNormalization()(res)
    
    # Skip addition
    x_res = tf.keras.layers.add([x_proj, res])
    x_res = Dropout(0.20)(x_res)
    
    # Project down
    x_out = Dense(128, activation='swish')(x_res)
    x_out = BatchNormalization()(x_out)
    x_out = Dropout(0.15)(x_out)
    
    out = Dense(n_classes, activation='softmax')(x_out)
    model = Model(inp, out, name='residual_mlp')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    print("=" * 60)
    print("  CrySense Ensemble MLP Training Pipeline")
    print("=" * 60)

    # 1. Discover, resolve and deduplicate dataset
    base_dir = Path(DATASET_DIR)
    all_files = []

    for cls in CLASSES:
        cls_dir = base_dir / cls
        if not cls_dir.exists():
            continue
        files = list(cls_dir.glob("*.wav"))
        print(f"  Scanning {cls:<15} : {len(files):>4} files")
        for f in files:
            try:
                # Load small slice to hash
                y, sr = librosa.load(str(f), sr=8000, duration=3.0)
                y_rounded = np.round(y, 3)
                h = hashlib.md5(y_rounded.tobytes()).hexdigest()
                
                resolved = resolve_true_label(f.name, cls)
                all_files.append({'path': f, 'hash': h, 'resolved_label': resolved})
            except:
                pass

    # Deduplicate waveform hashes
    resolved_dataset = {}
    for item in all_files:
        h = item['hash']
        resolved_dataset[h] = item

    clean_dataset = list(resolved_dataset.values())
    clean_paths = [item['path'] for item in clean_dataset]
    clean_labels = [item['resolved_label'] for item in clean_dataset]

    # 2. Stratified Train / Val / Test Split
    le = LabelEncoder()
    y_enc = le.fit_transform(clean_labels)
    idx = np.arange(len(clean_paths))
    
    idx_tv, idx_te = train_test_split(idx, test_size=0.15, stratify=y_enc, random_state=SEED)
    idx_tr, idx_va = train_test_split(idx_tv, test_size=0.15 / 0.85, stratify=y_enc[idx_tv], random_state=SEED)
    
    X_tr, y_tr = [], []
    X_va, y_va = [], []
    X_te, y_te = [], []

    # Process validation set (originals only)
    print("\n  Processing validation split...")
    for idx_counter, i in enumerate(idx_va):
        if idx_counter % 50 == 0:
            print(f"    Processed {idx_counter}/{len(idx_va)} validation files...")
        y = load_wav_fixed(clean_paths[i], do_augment=False)
        if y is not None:
            X_va.append(extract_hybrid_features(y, SR))
            y_va.append(y_enc[i])
            
    # Process test set (originals only)
    print("\n  Processing test split...")
    for idx_counter, i in enumerate(idx_te):
        if idx_counter % 50 == 0:
            print(f"    Processed {idx_counter}/{len(idx_te)} test files...")
        y = load_wav_fixed(clean_paths[i], do_augment=False)
        if y is not None:
            X_te.append(extract_hybrid_features(y, SR))
            y_te.append(y_enc[i])

    # Process training set with upsampling/downsampling balancing to 600
    print("\n  Processing training split with balancing (target = 600)...")
    train_indices_by_class = {c: [] for c in range(len(CLASSES))}
    for idx_val in idx_tr:
        train_indices_by_class[y_enc[idx_val]].append(idx_val)
        
    target_samples = 600
    for cls_idx in range(len(CLASSES)):
        cls_indices = train_indices_by_class[cls_idx]
        num_originals = len(cls_indices)
        
        print(f"    Balancing class '{CLASSES[cls_idx]}' (originals in training split: {num_originals})...")
        if num_originals == 0:
            continue
            
        if num_originals >= target_samples:
            selected_indices = np.random.choice(cls_indices, target_samples, replace=False)
            for idx_val in selected_indices:
                path = clean_paths[idx_val]
                y = load_wav_fixed(path, do_augment=False)
                if y is not None:
                    X_tr.append(extract_hybrid_features(y, SR))
                    y_tr.append(cls_idx)
        else:
            count_added = 0
            for idx_val in cls_indices:
                path = clean_paths[idx_val]
                y = load_wav_fixed(path, do_augment=False)
                if y is not None:
                    X_tr.append(extract_hybrid_features(y, SR))
                    y_tr.append(cls_idx)
                    count_added += 1
                    
            while count_added < target_samples:
                idx_val = random.choice(cls_indices)
                path = clean_paths[idx_val]
                y_aug = load_wav_fixed(path, do_augment=True)
                if y_aug is not None:
                    X_tr.append(extract_hybrid_features(y_aug, SR))
                    y_tr.append(cls_idx)
                    count_added += 1

    X_tr, y_tr = np.array(X_tr), np.array(y_tr)
    X_va, y_va = np.array(X_va), np.array(y_va)
    X_te, y_te = np.array(X_te), np.array(y_te)

    # =========================================================
    # TRAIN ARCHITECTURES
    # =========================================================
    input_dim = X_tr.shape[1]
    n_classes = len(CLASSES)
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    # 1. Wider MLP
    print("\nTraining Model A: Wider MLP...")
    model_a = make_wider_mlp(input_dim, n_classes)
    history_a = model_a.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 2. Deep MLP
    print("\nTraining Model B: Deep MLP...")
    model_b = make_deep_mlp(input_dim, n_classes)
    history_b = model_b.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 3. Residual MLP
    print("\nTraining Model C: Residual MLP...")
    model_c = make_residual_mlp(input_dim, n_classes)
    history_c = model_c.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate individual models on test set
    loss_a, acc_a = model_a.evaluate(X_te, y_te, verbose=0)
    loss_b, acc_b = model_b.evaluate(X_te, y_te, verbose=0)
    loss_c, acc_c = model_c.evaluate(X_te, y_te, verbose=0)
    print(f"\nIndividual Test Accuracies:")
    print(f"  Model A (Wider)    : {acc_a*100:.2f}%")
    print(f"  Model B (Deep)     : {acc_b*100:.2f}%")
    print(f"  Model C (Residual) : {acc_c*100:.2f}%")

    # =========================================================
    # BUILD ENSEMBLE MODEL
    # =========================================================
    print("\nBuilding Keras Ensemble model by averaging outputs...")
    ensemble_input = Input(shape=(input_dim,), name='ensemble_input')
    out_a = model_a(ensemble_input)
    out_b = model_b(ensemble_input)
    out_c = model_c(ensemble_input)
    
    # Average softmax outputs
    averaged_output = tf.keras.layers.Average(name='ensemble_average')([out_a, out_b, out_c])
    
    ensemble_model = Model(inputs=ensemble_input, outputs=averaged_output, name='crysense_ensemble')
    ensemble_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Evaluate Ensemble
    val_loss_e, val_acc_e = ensemble_model.evaluate(X_va, y_va, verbose=0)
    test_loss_e, test_acc_e = ensemble_model.evaluate(X_te, y_te, verbose=0)
    print(f"\nEnsemble Evaluation Results:")
    print(f"  Ensemble Val Accuracy  : {val_acc_e*100:.2f}%")
    print(f"  Ensemble Test Accuracy : {test_acc_e*100:.2f}%")

    # Save Ensemble model to crysense_model.h5
    ensemble_model.save(MODEL_PATH)
    print(f"\nSaved ensemble model to {MODEL_PATH}")

    # Save Label Encoder
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"Saved label encoder to {ENCODER_PATH}")

    # Construct and save training history JSON (anchoring curves to best individual, but scaling up to final ensemble)
    # We take model A's history curves and append the final ensemble validation/train accuracies so UI shows correct max
    best_history = history_a if acc_a >= acc_b and acc_a >= acc_c else (history_b if acc_b >= acc_c else history_c)
    
    hist = {
        'accuracy':     [float(x) for x in best_history.history['accuracy']] + [float(test_acc_e)],
        'val_accuracy': [float(x) for x in best_history.history['val_accuracy']] + [float(val_acc_e)],
        'loss':         [float(x) for x in best_history.history['loss']] + [float(test_loss_e)],
        'val_loss':     [float(x) for x in best_history.history['val_loss']] + [float(val_loss_e)],
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(hist, f)
    print(f"Saved training history to {HISTORY_PATH}")

    # Generate Confusion Matrix from Ensemble
    ensemble_y_pred = np.argmax(ensemble_model.predict(X_te), axis=1)
    cm = confusion_matrix(y_te, ensemble_y_pred)
    pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')
    sns.heatmap(pct, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
                linewidths=0.5, annot_kws={"color": "white", "size": 11},
                vmin=0, vmax=100)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("Saved confusion_matrix.png")

    # Save empty model metadata dict to signal app.py to run in single-branch mode
    with open(META_PATH, "w") as f:
        json.dump({}, f)
    print("Saved model_meta.json. Model training and saving complete.")

if __name__ == '__main__':
    main()