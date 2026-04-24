# -*- coding: utf-8 -*-
"""
CrySense - Baby Cry Detection ML Training Pipeline
====================================================
FIXED: Heavy oversampling + augmentation to handle class imbalance.
Classes: belly_pain, burping, discomfort, hungry, tired
"""

import os
import numpy as np
import librosa
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIRS   = ['dataset1', 'aug-dataset1']
CLASSES        = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
N_MFCC         = 40
MAX_LEN        = 128
EPOCHS         = 120
BATCH_SIZE     = 32
TARGET_SAMPLES = 300   # <-- oversample every class UP TO this count
MODEL_PATH     = 'crysense_model.h5'
ENCODER_PATH   = 'label_encoder.pkl'
HISTORY_PATH   = 'training_history.json'
SAMPLE_RATE    = 22050
DURATION       = 5


# ── Audio augmentation helpers ────────────────────────────────────────────────

def augment_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply one random augmentation to a waveform."""
    choice = np.random.randint(0, 5)
    if choice == 0:
        # time stretch
        rate = np.random.uniform(0.8, 1.2)
        y = librosa.effects.time_stretch(y, rate=rate)
    elif choice == 1:
        # pitch shift
        steps = np.random.randint(-3, 4)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    elif choice == 2:
        # add gaussian noise
        noise = np.random.normal(0, 0.005, len(y))
        y = y + noise
    elif choice == 3:
        # shift in time
        shift = np.random.randint(sr // 4, sr)
        y = np.roll(y, shift)
    else:
        # change volume
        y = y * np.random.uniform(0.7, 1.3)
    return y


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract MFCC + delta + delta-delta features."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta1, delta2])  # (120, T)

    if combined.shape[1] < MAX_LEN:
        combined = np.pad(combined, ((0, 0), (0, MAX_LEN - combined.shape[1])), mode='constant')
    else:
        combined = combined[:, :MAX_LEN]

    combined = (combined - combined.mean()) / (combined.std() + 1e-8)
    return combined


def load_raw_audio(file_path: str):
    """Load raw waveform."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        return y, sr
    except Exception as e:
        print(f"  [WARN] Could not load {file_path}: {e}")
        return None, None


# ── Dataset loading with oversampling ─────────────────────────────────────────

def load_dataset():
    """Load all audio, oversample minority classes, return features."""
    # Step 1: collect raw waveforms per class
    raw_by_class = {cls: [] for cls in CLASSES}

    for dataset_dir in DATASET_DIRS:
        base = Path(dataset_dir)
        if not base.exists():
            print(f"  [SKIP] {dataset_dir} not found")
            continue
        for cls in CLASSES:
            cls_dir = base / cls
            if not cls_dir.exists():
                continue
            files = list(cls_dir.glob('*.wav'))
            print(f"  {dataset_dir}/{cls}: {len(files)} files")
            for fp in files:
                y, sr = load_raw_audio(str(fp))
                if y is not None:
                    raw_by_class[cls].append((y, sr))

    print("\n  --- Class counts before oversampling ---")
    for cls in CLASSES:
        print(f"  {cls}: {len(raw_by_class[cls])}")

    # Step 2: oversample minority classes by augmentation
    X, labels = [], []
    for cls in CLASSES:
        samples = raw_by_class[cls]
        if len(samples) == 0:
            print(f"  [WARN] No samples for class {cls}!")
            continue

        # Add original samples
        for y, sr in samples:
            feat = extract_features(y, sr)
            X.append(feat)
            labels.append(cls)

        # Oversample up to TARGET_SAMPLES
        needed = TARGET_SAMPLES - len(samples)
        if needed > 0:
            print(f"  Augmenting {cls}: generating {needed} extra samples ...")
            for i in range(needed):
                y, sr = samples[i % len(samples)]
                y_aug = augment_audio(y.copy(), sr)
                feat = extract_features(y_aug, sr)
                X.append(feat)
                labels.append(cls)

    print("\n  --- Class counts after oversampling ---")
    for cls in CLASSES:
        count = labels.count(cls)
        print(f"  {cls}: {count}")

    return np.array(X), np.array(labels)


# ── Model Architecture ─────────────────────────────────────────────────────────

def build_cnn(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """Deeper CNN with GlobalAveragePooling instead of Flatten to reduce overfitting."""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),

        # Use GlobalAveragePooling instead of Flatten → less overfitting
        GlobalAveragePooling2D(),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Plot helpers ───────────────────────────────────────────────────────────────

def plot_history(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f0f1a')
    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.spines[:].set_color('#444')
        ax.tick_params(colors='#ccc')
        ax.yaxis.label.set_color('#ccc')
        ax.xaxis.label.set_color('#ccc')
        ax.title.set_color('#fff')

    axes[0].plot(history['accuracy'],     color='#6c63ff', lw=2, label='Train Acc')
    axes[0].plot(history['val_accuracy'], color='#ff6584', lw=2, label='Val Acc')
    axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(facecolor='#1a1a2e', labelcolor='white')

    axes[1].plot(history['loss'],     color='#6c63ff', lw=2, label='Train Loss')
    axes[1].plot(history['val_loss'], color='#ff6584', lw=2, label='Val Loss')
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(facecolor='#1a1a2e', labelcolor='white')

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  Saved training_curves.png")


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=classes, yticklabels=classes,
                ax=ax, linewidths=0.5, annot_kws={"color": "white"})
    ax.set_title('Confusion Matrix', color='white', fontsize=14, pad=12)
    ax.set_xlabel('Predicted', color='#ccc')
    ax.set_ylabel('Actual', color='#ccc')
    ax.tick_params(colors='#ccc')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("  Saved confusion_matrix.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  CrySense — Baby Cry Detection Training (FIXED)")
    print("="*60)

    # 1. Load data with oversampling
    print("\n[1/5] Loading & oversampling dataset …")
    X, y_raw = load_dataset()
    print(f"\n  Total samples after oversampling: {len(X)}")
    if len(X) == 0:
        print("  ERROR: No audio files found. Check dataset dirs.")
        return

    # 2. Encode labels
    print("\n[2/5] Encoding labels …")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    y_cat = to_categorical(y_enc)
    print(f"  Classes: {list(le.classes_)}")

    # 3. Train / val split — stratified so each split is balanced
    X = X[..., np.newaxis]   # (N, 120, 128, 1) for Conv2D
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}")

    # 4. Build & train
    print("\n[3/5] Building CNN model ...")
    model = build_cnn(X_train.shape[1:], len(le.classes_))
    model.summary()

    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7, verbose=1)
    ]

    print(f"\n[4/5] Training for up to {EPOCHS} epochs ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Evaluate & save
    print("\n[5/5] Evaluating …")
    y_pred_prob = model.predict(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_val, axis=1)

    print("\n" + classification_report(y_true, y_pred, target_names=le.classes_))

    plot_history(history.history)
    plot_confusion_matrix(y_true, y_pred, le.classes_)

    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"  Saved label encoder -> {ENCODER_PATH}")

    hist_json = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    with open(HISTORY_PATH, 'w') as f:
        json.dump(hist_json, f)
    print(f"  Saved training history -> {HISTORY_PATH}")

    val_acc = max(history.history['val_accuracy'])
    print(f"\n  [OK] Best Val Accuracy: {val_acc*100:.2f}%")
    print(f"  [OK] Model saved -> {MODEL_PATH}")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
