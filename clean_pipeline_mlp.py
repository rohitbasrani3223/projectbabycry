import os, random, json, pickle, hashlib, warnings
os.environ["TFHUB_CACHE_DIR"] = os.path.abspath("./tfhub_modules")
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Config
DATASET_DIR = "aug-dataset1"
CLASSES = ["belly_pain", "burping", "cold_hot", "discomfort", "hungry", "tired"]
SR = 16000
DURATION = 7
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
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
        y_8k, sr_8k = librosa.load(str(path), sr=8000, duration=DURATION, mono=True)
        if len(y_8k) < 8000 * 0.5:
            return None
        y = librosa.resample(y_8k, orig_sr=8000, target_sr=16000)
        y = clean_audio(y)
        if do_augment:
            y = augment_audio(y, SR)
            y = clean_audio(y)
        target_len = SR * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]
        return y.astype(np.float32)
    except:
        return None

def extract_hybrid_features(y, sr):
    # 1. YAMNet
    scores, embeddings, spectrogram = yamnet_model(y)
    yamnet_emb = embeddings.numpy()
    yamnet_mean = np.mean(yamnet_emb, axis=0)
    yamnet_max = np.max(yamnet_emb, axis=0)
    yamnet_features = np.concatenate([yamnet_mean, yamnet_max])
    
    # 2. Traditional fine-grained features
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
    
    trad_features = np.concatenate([
        mfcc_mean, mfcc_std,
        [centroid_mean, centroid_std],
        contrast_mean, contrast_std,
        [zcr_mean, zcr_std],
        [rms_mean, rms_std]
    ])
    
    return np.concatenate([yamnet_features, trad_features])

# Resolve & clean
print("Scanning and cleansing dataset...")
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

resolved_dataset = {}
for item in all_files:
    h = item['hash']
    resolved_dataset[h] = item

clean_dataset = list(resolved_dataset.values())
print(f"Total unique waveforms: {len(clean_dataset)}")

clean_paths = [item['path'] for item in clean_dataset]
clean_labels = [item['resolved_label'] for item in clean_dataset]

# Split
le = LabelEncoder()
y_enc = le.fit_transform(clean_labels)
idx = np.arange(len(clean_paths))

idx_tv, idx_te = train_test_split(idx, test_size=0.15, stratify=y_enc, random_state=SEED)
idx_tr, idx_va = train_test_split(idx_tv, test_size=0.15 / 0.85, stratify=y_enc[idx_tv], random_state=SEED)

print(f"Train size: {len(idx_tr)} | Val size: {len(idx_va)} | Test size: {len(idx_te)}")

# Extract features
X_tr, y_tr = [], []
X_va, y_va = [], []
X_te, y_te = [], []

print("\nExtracting validation and test features...")
for i in idx_va:
    y = load_wav_fixed(clean_paths[i], do_augment=False)
    if y is not None:
        X_va.append(extract_hybrid_features(y, SR))
        y_va.append(y_enc[i])

for i in idx_te:
    y = load_wav_fixed(clean_paths[i], do_augment=False)
    if y is not None:
        X_te.append(extract_hybrid_features(y, SR))
        y_te.append(y_enc[i])

print("\nExtracting balanced training features...")
train_indices_by_class = {c: [] for c in range(len(CLASSES))}
for idx_val in idx_tr:
    train_indices_by_class[y_enc[idx_val]].append(idx_val)

target_samples = 400
for cls_idx in range(len(CLASSES)):
    cls_name = le.classes_[cls_idx]
    cls_indices = train_indices_by_class[cls_idx]
    num_originals = len(cls_indices)
    if num_originals == 0:
        continue
    multiplier = max(1, int(np.ceil(target_samples / num_originals)))
    print(f"  Class {cls_name:<12} (originals: {num_originals:>3}) -> multiplier {multiplier}x")
    
    for idx_val in cls_indices:
        path = clean_paths[idx_val]
        # Clean
        y = load_wav_fixed(path, do_augment=False)
        if y is not None:
            X_tr.append(extract_hybrid_features(y, SR))
            y_tr.append(cls_idx)
        # Augment
        for _ in range(multiplier - 1):
            y_aug = load_wav_fixed(path, do_augment=True)
            if y_aug is not None:
                X_tr.append(extract_hybrid_features(y_aug, SR))
                y_tr.append(cls_idx)

X_tr, y_tr = np.array(X_tr), np.array(y_tr)
X_va, y_va = np.array(X_va), np.array(y_va)
X_te, y_te = np.array(X_te), np.array(y_te)

print(f"\nFeatures Extracted:")
print(f"  X_tr: {X_tr.shape} | X_va: {X_va.shape} | X_te: {X_te.shape}")

# Model
def build_simple_mlp(input_dim, n_classes):
    inp = Input(shape=(input_dim,))
    
    x = BatchNormalization()(inp)
    x = Dropout(0.25)(x)
    
    x = Dense(128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.20)(x)
    
    x = Dense(64, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    
    out = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_simple_mlp(X_tr.shape[1], len(CLASSES))
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
print(f"\nFinal Genuine stratified Test Accuracy: {test_acc*100:.2f}%")
y_pred = np.argmax(model.predict(X_te), axis=1)
print(classification_report(y_te, y_pred, target_names=le.classes_))
