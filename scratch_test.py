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

def clean_audio(y):
    return librosa.util.normalize(y)

def extract_hybrid_features(y, sr):
    scores, embeddings, spectrogram = yamnet_model(y)
    yamnet_emb = embeddings.numpy()
    yamnet_mean = np.mean(yamnet_emb, axis=0)
    yamnet_max = np.max(yamnet_emb, axis=0)
    yamnet_features = np.concatenate([yamnet_mean, yamnet_max])
    
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

def load_wav_fixed(path):
    try:
        y_8k, sr_8k = librosa.load(str(path), sr=8000, duration=DURATION, mono=True)
        if len(y_8k) < 8000 * 0.5:
            return None
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

# First, scan and load waveforms to find cross-class duplicates
print("Scanning dataset to remove cross-class duplicates...")
base_dir = Path(DATASET_DIR)
all_wavs_data = []

# To find cross-class duplicates
waveform_hashes = {}
duplicate_wav_hashes = set()

for cls in CLASSES:
    wavs = list((base_dir / cls).glob("*.wav"))
    for p in wavs:
        # Load briefly at 8k to get a waveform hash
        try:
            y, sr = librosa.load(str(p), sr=8000, duration=DURATION)
            y_rounded = np.round(y, 3)
            h = hashlib.md5(y_rounded.tobytes()).hexdigest()
            
            if h in waveform_hashes:
                other_class = waveform_hashes[h]
                if other_class != cls:
                    duplicate_wav_hashes.add(h)
            else:
                waveform_hashes[h] = cls
                
            all_wavs_data.append({'path': p, 'label': cls, 'hash': h})
        except:
            pass

print(f"Total wav files loaded: {len(all_wavs_data)}")
print(f"Total unique waveform hashes: {len(waveform_hashes)}")
print(f"Cross-class duplicate hashes to discard: {len(duplicate_wav_hashes)}")

# Filter out all files that share duplicate hashes across different classes
clean_wavs_data = [item for item in all_wavs_data if item['hash'] not in duplicate_wav_hashes]
print(f"Remaining clean, uniquely-labeled wav files: {len(clean_wavs_data)}")

# Take subset for quick diagnosis (up to 80 files per class from clean ones)
class_counts = {cls: 0 for cls in CLASSES}
filtered_clean_wavs = []
for item in clean_wavs_data:
    cls = item['label']
    if class_counts[cls] < 80:
        filtered_clean_wavs.append(item)
        class_counts[cls] += 1

print("Filtered clean class distribution:")
for cls, count in class_counts.items():
    print(f"  {cls}: {count} files")

# Extract features
le = LabelEncoder()
y_labels = [item['label'] for item in filtered_clean_wavs]
y_enc = le.fit_transform(y_labels)

X_hybrid = []
y_valid = []
print("\nExtracting features from clean dataset...")
for count, item in enumerate(filtered_clean_wavs):
    y = load_wav_fixed(item['path'])
    if y is not None:
        feat = extract_hybrid_features(y, SR)
        X_hybrid.append(feat)
        y_valid.append(y_enc[count])
    if (count+1) % 50 == 0:
        print(f"  Processed {count+1}/{len(filtered_clean_wavs)}")

X_hybrid = np.array(X_hybrid)
y_valid = np.array(y_valid)

print(f"Features shape: {X_hybrid.shape}")

# Split and train a robust classifier
X_train, X_test, y_train, y_test = train_test_split(
    X_hybrid, y_valid, test_size=0.2, stratify=y_valid, random_state=SEED
)

clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=SEED, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== DEDUPLICATED, UNBIASED Random Forest Diagnostic Results ===")
print(f"Stratified Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=le.classes_))
