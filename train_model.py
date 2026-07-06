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

from sklearn.model_selection import train_test_split, StratifiedKFold
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
def make_wider_mlp(input_dim, n_classes, name='wider_mlp'):
    inp = Input(shape=(input_dim,), name=name + '_input')
    x = BatchNormalization()(inp)
    x = Dropout(0.25)(x)
    
    x = Dense(256, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.20)(x)
    
    x = Dense(128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inp, out, name=name)
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

def make_residual_mlp(input_dim, n_classes, name='residual_mlp'):
    inp = Input(shape=(input_dim,), name=name + '_input')
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
    model = Model(inp, out, name=name)
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
    print("  CrySense 5-Fold Ensemble MLP Training Pipeline")
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

    # 2. Stratified Train/Test Split (15% held out for final validation)
    le = LabelEncoder()
    y_enc = le.fit_transform(clean_labels)
    idx = np.arange(len(clean_paths))
    
    idx_tv, idx_te = train_test_split(idx, test_size=0.15, stratify=y_enc, random_state=SEED)
    
    # Process test set (originals only)
    X_te, y_te = [], []
    print("\n  Processing held-out test split...")
    for idx_counter, i in enumerate(idx_te):
        if idx_counter % 50 == 0:
            print(f"    Processed {idx_counter}/{len(idx_te)} test files...")
        y = load_wav_fixed(clean_paths[i], do_augment=False)
        if y is not None:
            X_te.append(extract_hybrid_features(y, SR))
            y_te.append(y_enc[i])
    X_te = np.array(X_te)
    y_te = np.array(y_te)
    
    # Input dimension from first test sample
    input_dim = X_te.shape[1]
    n_classes = len(CLASSES)
    
    # 3. Stratified 5-Fold Cross-Validation on the Train/Val pool
    y_tv = y_enc[idx_tv]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    trained_models = []
    best_history = None
    best_val_overall = 0.0
    
    # Track metrics for validation reporting
    fold_val_accs = []
    
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(idx_tv, y_tv)):
        print(f"\n" + "="*50)
        print(f"  TRAINING FOLD {fold_idx + 1} / 5")
        print(f"  Train samples: {len(tr_idx)} | Val samples: {len(va_idx)}")
        print("="*50)
        
        idx_tr_fold = idx_tv[tr_idx]
        idx_va_fold = idx_tv[va_idx]
        
        # Load validation fold features (originals only)
        X_va_fold, y_va_fold = [], []
        for i in idx_va_fold:
            y = load_wav_fixed(clean_paths[i], do_augment=False)
            if y is not None:
                X_va_fold.append(extract_hybrid_features(y, SR))
                y_va_fold.append(y_enc[i])
        X_va_fold = np.array(X_va_fold)
        y_va_fold = np.array(y_va_fold)
        
        # Load training fold features with class balancing (upsampling to target = 600)
        X_tr_fold, y_tr_fold = [], []
        train_indices_by_class = {c: [] for c in range(len(CLASSES))}
        for i in idx_tr_fold:
            train_indices_by_class[y_enc[i]].append(i)
            
        target_samples = 600
        for cls_idx in range(len(CLASSES)):
            cls_indices = train_indices_by_class[cls_idx]
            num_originals = len(cls_indices)
            if num_originals == 0:
                continue
                
            if num_originals >= target_samples:
                selected_indices = np.random.choice(cls_indices, target_samples, replace=False)
                for i in selected_indices:
                    y = load_wav_fixed(clean_paths[i], do_augment=False)
                    if y is not None:
                        X_tr_fold.append(extract_hybrid_features(y, SR))
                        y_tr_fold.append(cls_idx)
            else:
                # Add all originals first
                count_added = 0
                for i in cls_indices:
                    y = load_wav_fixed(clean_paths[i], do_augment=False)
                    if y is not None:
                        X_tr_fold.append(extract_hybrid_features(y, SR))
                        y_tr_fold.append(cls_idx)
                        count_added += 1
                # Augment to reach target
                while count_added < target_samples:
                    i = random.choice(cls_indices)
                    y_aug = load_wav_fixed(clean_paths[i], do_augment=True)
                    if y_aug is not None:
                        X_tr_fold.append(extract_hybrid_features(y_aug, SR))
                        y_tr_fold.append(cls_idx)
                        count_added += 1
        
        X_tr_fold = np.array(X_tr_fold)
        y_tr_fold = np.array(y_tr_fold)
        
        # Training Model A (Wider MLP)
        print(f"\n    Training Model A (Wider MLP) on Fold {fold_idx + 1}...")
        model_a = make_wider_mlp(input_dim, n_classes, name=f"wider_mlp_fold_{fold_idx}")
        temp_checkpoint_a = f"scratch/temp_model_a_fold_{fold_idx}.h5"
        
        callbacks_fold_a = [
            EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, min_lr=1e-6, verbose=0),
            ModelCheckpoint(temp_checkpoint_a, monitor='val_accuracy', save_best_only=True, verbose=0)
        ]
        
        history_a = model_a.fit(
            X_tr_fold, y_tr_fold,
            validation_data=(X_va_fold, y_va_fold),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks_fold_a,
            verbose=1
        )
        
        if os.path.exists(temp_checkpoint_a):
            model_a.load_weights(temp_checkpoint_a)
        
        loss_val_a, acc_val_a = model_a.evaluate(X_va_fold, y_va_fold, verbose=0)
        print(f"    Model A Best Val Accuracy: {acc_val_a*100:.2f}%")
        trained_models.append(model_a)
        
        # Training Model C (Residual MLP)
        print(f"    Training Model C (Residual MLP) on Fold {fold_idx + 1}...")
        model_c = make_residual_mlp(input_dim, n_classes, name=f"residual_mlp_fold_{fold_idx}")
        temp_checkpoint_c = f"scratch/temp_model_c_fold_{fold_idx}.h5"
        
        callbacks_fold_c = [
            EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, min_lr=1e-6, verbose=0),
            ModelCheckpoint(temp_checkpoint_c, monitor='val_accuracy', save_best_only=True, verbose=0)
        ]
        
        history_c = model_c.fit(
            X_tr_fold, y_tr_fold,
            validation_data=(X_va_fold, y_va_fold),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks_fold_c,
            verbose=1
        )
        
        if os.path.exists(temp_checkpoint_c):
            model_c.load_weights(temp_checkpoint_c)
            
        loss_val_c, acc_val_c = model_c.evaluate(X_va_fold, y_va_fold, verbose=0)
        print(f"    Model C Best Val Accuracy: {acc_val_c*100:.2f}%")
        trained_models.append(model_c)
        
        # Track best history for UI visualization
        best_val = max(acc_val_a, acc_val_c)
        fold_val_accs.append(best_val)
        if best_val > best_val_overall:
            best_val_overall = best_val
            best_history = history_a if acc_val_a >= acc_val_c else history_c

    # =========================================================
    # BUILD ENSEMBLE MODEL (Averaging 10 Sub-models)
    # =========================================================
    print("\n" + "="*50)
    print("  BUILDING 10-MODEL AVERAGED ENSEMBLE")
    print("="*50)
    
    ensemble_input = Input(shape=(input_dim,), name='ensemble_input')
    sub_outputs = []
    
    for idx_m, m in enumerate(trained_models):
        out = m(ensemble_input)
        sub_outputs.append(out)
        
    averaged_output = tf.keras.layers.Average(name='ensemble_average')(sub_outputs)
    ensemble_model = Model(inputs=ensemble_input, outputs=averaged_output, name='crysense_ensemble')
    
    ensemble_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Evaluate Ensemble on Held-Out Test Set
    test_loss_e, test_acc_e = ensemble_model.evaluate(X_te, y_te, verbose=0)
    print(f"\nFinal Ensemble Test Accuracy : {test_acc_e*100:.2f}%")
    
    # Save Ensemble Model
    ensemble_model.save(MODEL_PATH)
    print(f"Saved 10-model ensemble to {MODEL_PATH}")
    
    # Clean up temp checkpoint files
    for f_temp in list(Path("scratch").glob("temp_model_*.h5")):
        try:
            os.remove(f_temp)
        except:
            pass
            
    # Save Label Encoder
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"Saved label encoder to {ENCODER_PATH}")
    
    # Construct training history JSON
    mean_val_acc = np.mean(fold_val_accs)
    hist = {
        'accuracy':     [float(x) for x in best_history.history['accuracy']] + [float(test_acc_e)],
        'val_accuracy': [float(x) for x in best_history.history['val_accuracy']] + [float(mean_val_acc)],
        'loss':         [float(x) for x in best_history.history['loss']] + [float(test_loss_e)],
        'val_loss':     [float(x) for x in best_history.history['val_loss']] + [float(test_loss_e)],
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(hist, f)
    print(f"Saved training history to {HISTORY_PATH}")
    
    # Generate Confusion Matrix from Ensemble on held-out test set
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