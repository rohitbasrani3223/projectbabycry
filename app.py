"""
CrySense - Flask Backend API
==============================
Endpoints:
  POST /predict        - Upload .wav file → get cry type prediction
  GET  /model-info     - Model accuracy, classes, training stats
  GET  /history        - Training history JSON
  GET  /health         - Server health
"""

import os
# Set TFHUB cache directory in workspace before any tensorflow imports
os.environ["TFHUB_CACHE_DIR"] = os.path.abspath("./tfhub_modules")

import json
import pickle
import tempfile
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

import subprocess
import sys
import shutil
import matplotlib
matplotlib.use('Agg') # Headless backend for Flask threads
import matplotlib.pyplot as plt
import librosa.display

# ── Model Meta (dual-branch config) ───────────────────────────────────────────
MODEL_META_PATH = 'model_meta.json'
_meta = {}
if os.path.exists(MODEL_META_PATH):
    with open(MODEL_META_PATH) as f:
        _meta = json.load(f)

N_MELS      = _meta.get('n_mels', 128)
TIME_FRAMES = _meta.get('time_frames', 215)
N_FFT       = _meta.get('n_fft', 2048)
HOP_LENGTH  = _meta.get('hop_length', 512)
IS_DUAL     = bool(_meta)  # True if v18+ dual-branch model

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = 'crysense_model.h5'
ENCODER_PATH = 'label_encoder.pkl'
HISTORY_PATH = 'training_history.json'

app = Flask(__name__, static_folder='static')
CORS(app)

model            = None
label_enc        = None
training_history = {}

print("Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet loaded.")
# ── Rich Emotion Data ──────────────────────────────────────────────────────────
CRY_INFO = {
    'belly_pain': {
        'emoji':       '🤢',
        'color':       '#ef4444',
        'severity':    'high',
        'label':       'Belly Pain / Gas',
        'advice':      'Baby may have gas or stomach cramps. Try gentle tummy massage in slow clockwise circles, bicycle leg movements, or hold baby over your shoulder with gentle back pats.',
        'advice_hi':   'बच्चे को गैस या पेट दर्द हो सकता है। पेट पर धीरे-धीरे घड़ी की दिशा में मालिश करें, साइकिल की तरह पैर हिलाएं, या कंधे पर उठाकर पीठ थपथपाएं।',
        'quick_tips':  [
            '🔄 Clockwise tummy massage',
            '🚲 Bicycle leg movements',
            '🤱 Hold upright on shoulder',
            '🌡️ Check for fever',
            '🍼 Check if overfeeding'
        ],
        'secondary_emotions': ['discomfort', 'scared'],
        'doctor_warning': 'If crying persists over 3 hours, baby has fever, or vomiting — consult doctor immediately.',
        'duration_tip': 'Usually resolves in 20–40 minutes with proper soothing.'
    },
    'burping': {
        'emoji':       '💨',
        'color':       '#f97316',
        'severity':    'low',
        'label':       'Needs to Burp',
        'advice':      'Baby has trapped air after feeding. Hold baby upright and gently pat or rub their back in an upward motion until they burp. Try the over-the-shoulder or face-down-on-lap positions.',
        'advice_hi':   'दूध पीने के बाद बच्चे को डकार दिलानी है। बच्चे को सीधा पकड़ें और पीठ पर धीरे-धीरे थपथपाएं। कंधे पर उठाकर या पेट के बल घुटने पर लिटाकर भी कोशिश करें।',
        'quick_tips':  [
            '🤱 Hold upright on shoulder',
            '👐 Gentle circular back rub',
            '🦵 Face-down on lap method',
            '⏰ Pat for at least 2–3 minutes',
            '🍼 Feed more slowly next time'
        ],
        'secondary_emotions': ['discomfort', 'hungry'],
        'doctor_warning': 'If baby spits up excessively or seems in pain after every feeding, check for reflux.',
        'duration_tip': 'Usually resolves within 5 minutes of proper burping technique.'
    },
    'discomfort': {
        'emoji':       '😣',
        'color':       '#eab308',
        'severity':    'medium',
        'label':       'Discomfort / Irritation',
        'advice':      'Baby is uncomfortable. Check for wet or soiled diaper, clothing that is too tight, room temperature issues, skin rash or irritation, or anything poking/scratching the baby.',
        'advice_hi':   'बच्चा असहज है। गीला या गंदा डायपर चेक करें, कपड़े बहुत टाइट तो नहीं, कमरे का तापमान सही है या नहीं, त्वचा पर दाने तो नहीं हैं।',
        'quick_tips':  [
            '👶 Check & change diaper',
            '🌡️ Check room temperature (20–22°C ideal)',
            '👕 Loosen tight clothing',
            '🔍 Check for hair tourniquet on fingers/toes',
            '💆 Gentle skin-to-skin contact'
        ],
        'secondary_emotions': ['tired', 'belly_pain'],
        'doctor_warning': 'If you find a rash, swelling, or the baby cannot be soothed — see a pediatrician.',
        'duration_tip': 'Once the cause is fixed, baby should calm down within 5–10 minutes.'
    },
    'hungry': {
        'emoji':       '🍼',
        'color':       '#22c55e',
        'severity':    'medium',
        'label':       'Hungry / Needs Feeding',
        'advice':      'Baby is hungry! Feed immediately. Look for early hunger cues: rooting reflex, sucking hands, turning head side to side. Do not wait until the cry is intense as it makes latching harder.',
        'advice_hi':   'बच्चे को भूख लगी है! तुरंत दूध पिलाएं। भूख के शुरुआती संकेत: मुंह घुमाना, हाथ चूसना, सिर इधर-उधर करना। रोने तक इंतज़ार न करें।',
        'quick_tips':  [
            '🤱 Breastfeed or offer bottle immediately',
            '👀 Check rooting reflex signs',
            '⏰ Track feeding schedule (every 2–3 hrs newborn)',
            '📊 Monitor wet diapers (6+ per day = well fed)',
            '🌙 Night feeds are normal for infants'
        ],
        'secondary_emotions': ['tired', 'discomfort'],
        'doctor_warning': 'If baby feeds but still seems unsatisfied, check milk supply or formula preparation.',
        'duration_tip': 'Newborns feed every 1.5–3 hours. Older babies every 3–4 hours.'
    },
    'tired': {
        'emoji':       '😴',
        'color':       '#6c63ff',
        'severity':    'low',
        'label':       'Tired / Sleepy',
        'advice':      'Baby is overtired and needs sleep. Create a calm environment: dim lights, reduce noise, use white noise, try rocking or swaddling. Watch for sleep cues: eye rubbing, yawning, staring blankly.',
        'advice_hi':   'बच्चा थका हुआ है और सोना चाहता है। शांत माहौल बनाएं: रोशनी कम करें, शोर कम करें, सफेद आवाज़ चलाएं, झुलाएं या कपड़े में लपेटें।',
        'quick_tips':  [
            '🌙 Dim the lights',
            '🔇 Reduce noise & stimulation',
            '🌊 Play white noise / fan sound',
            '🤗 Swaddle snugly',
            '🪂 Gentle rhythmic rocking'
        ],
        'secondary_emotions': ['discomfort', 'hungry'],
        'doctor_warning': 'If baby has difficulty sleeping consistently or seems lethargic during awake time — consult doctor.',
        'duration_tip': 'Newborns sleep 16–18 hrs/day. Create a bedtime routine from early on.'
    },
    'cold_hot': {
        'emoji':       '🌡️',
        'color':       '#06b6d4',
        'severity':    'high',
        'label':       'Too Cold / Too Hot',
        'advice':      'Baby is uncomfortable due to temperature — either too cold or too hot. Check the room temperature (ideal 20–22°C), feel baby\'s chest/back (not hands/feet) to gauge body temperature, and adjust clothing or blankets accordingly.',
        'advice_hi':   'बच्चे को ठंड या गर्मी लग रही है। कमरे का तापमान जाँचें (20–22°C आदर्श है), बच्चे की छाती या पीठ छूकर देखें, और कपड़े या कंबल ठीक करें।',
        'quick_tips':  [
            '🌡️ Check room temperature (20–22°C ideal)',
            '👕 Add/remove a layer of clothing',
            '🤚 Feel chest/back — not hands/feet',
            '🪟 Ventilate the room if too hot',
            '🧣 Use a light blanket if too cold'
        ],
        'secondary_emotions': ['discomfort', 'tired'],
        'doctor_warning': 'If  A baby feels very hot (fever > 38°C / 100.4°F) or very cold and unresponsive — seek medical help immediately.',
        'duration_tip': 'Once temperature is adjusted, baby should settle within 5–10 minutes.'
    }
}

# ── Secondary/Inferred Emotions ────────────────────────────────────────────────
# These are rule-based inferences BEYOND the model's 5 classes
INFERRED_EMOTIONS = {
    'colic': {
        'emoji':    '😭',
        'color':    '#dc2626',
        'label':    'Possible Colic',
        'desc':     'High-intensity crying that may indicate colic — intense, inconsolable crying in an otherwise healthy baby, often in the evenings.',
        'desc_hi':  'बहुत तेज़ और लंबे समय तक रोना कोलिक का संकेत हो सकता है।',
        'tip':      'Rule of 3: Colic is crying 3+ hrs/day, 3+ days/week, for 3+ weeks. Consult pediatrician.',
    },
    'scared': {
        'emoji':    '😨',
        'color':    '#7c3aed',
        'label':    'Possibly Scared / Startled',
        'desc':     'Baby may have been startled or is feeling anxious. Sudden loud noises, unfamiliar faces, or abrupt movements can trigger this cry.',
        'desc_hi':  'बच्चा डरा हुआ या चौंका हुआ हो सकता है। तेज़ आवाज़ या अचानक हलचल से डर लग सकता है।',
        'tip':      'Hold baby close, speak softly, and remove the source of fright.',
    },
    'lonely': {
        'emoji':    '🥺',
        'color':    '#0284c7',
        'label':    'Lonely / Needs Attention',
        'desc':     'Baby wants to be held or needs human interaction. Babies cry for connection — this is completely normal and healthy.',
        'desc_hi':  'बच्चा गोद में आना चाहता है या ध्यान चाहता है। यह बिल्कुल सामान्य है।',
        'tip':      'Pick baby up, make eye contact, talk or sing softly. Cuddles are never "spoiling".',
    },
    'overstimulated': {
        'emoji':    '🌀',
        'color':    '#0891b2',
        'label':    'Overstimulated',
        'desc':     'Baby has received too much sensory input — bright lights, noise, many people, or too much activity. They need a calm, quiet space to reset.',
        'desc_hi':  'बच्चे को बहुत ज़्यादा उत्तेजना मिल गई — तेज़ रोशनी, शोर, भीड़। उसे शांत जगह चाहिए।',
        'tip':      'Take baby to a quiet, dark room. Reduce all stimulation. Gentle swaying helps.',
    },
    'teething': {
        'emoji':    '🦷',
        'color':    '#b45309',
        'label':    'Possibly Teething Pain',
        'desc':     'If baby is 4–7 months old, teething could be causing pain and discomfort. Look for drooling, chewing on hands, and swollen gums.',
        'desc_hi':  '4–7 महीने की उम्र में दांत निकलने की तकलीफ हो सकती है। लार ज़्यादा आना, हाथ चबाना संकेत हैं।',
        'tip':      'Use a cold (not frozen) teething ring. Gently rub gums with clean finger.',
    }
}


def infer_secondary_emotion(prediction: str, confidence: float, all_probs: dict) -> dict | None:
    """
    Rule-based secondary emotion inference from model output patterns.
    Returns an inferred emotion dict or None.
    """
    # Very low confidence = scared or overstimulated
    if confidence < 45:
        return INFERRED_EMOTIONS['scared']

    # belly_pain + high confidence = possibly colic
    if prediction == 'belly_pain' and confidence > 65:
        # Check if it's sustained (we can't know duration, so flag it)
        return INFERRED_EMOTIONS['colic']

    # tired + low discomfort but medium confidence = overstimulated
    if prediction == 'tired' and confidence < 70:
        tired_p   = all_probs.get('tired', 0)
        discomf_p = all_probs.get('discomfort', 0)
        if discomf_p > 15:
            return INFERRED_EMOTIONS['overstimulated']

    # discomfort + low confidence + low hungry = lonely
    if prediction == 'discomfort' and confidence < 60:
        hungry_p = all_probs.get('hungry', 0)
        if hungry_p < 20:
            return INFERRED_EMOTIONS['lonely']

    # burping but high belly_pain prob = possibly teething if older
    if prediction == 'burping':
        bp_p = all_probs.get('belly_pain', 0)
        if bp_p > 20:
            return INFERRED_EMOTIONS['teething']

    return None


# ── Load Resources ─────────────────────────────────────────────────────────────
def load_resources():
    global model, label_enc, training_history
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[OK] Model loaded from {MODEL_PATH}")
        else:
            print(f"[WARN] Model not found — run train_model.py first")
    except Exception as e:
        print(f"[ERR] Error loading model: {e}")

    # Load label encoder
    try:
        if os.path.exists(ENCODER_PATH):
            with open(ENCODER_PATH, 'rb') as f:
                label_enc = pickle.load(f)
            print("[OK] Label encoder loaded")
    except Exception as e:
        print(f"[ERR] Error loading encoder: {e}")

    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r') as f:
                training_history = json.load(f)
    except:
        pass


# ── Feature Extraction (v18 Dual-Branch: CNN + YAMNet) ─────────────────────────
SR       = 16000
DURATION = 7

def clean_audio(y):
    return librosa.util.normalize(y)

def _load_uniform(file_path):
    """Load audio directly at 16kHz to preserve high-frequency crying details."""
    y, _ = librosa.load(file_path, sr=16000, duration=DURATION, mono=True)
    if len(y) < 16000 * 0.5:
        raise ValueError("Audio clip is too short or empty")
    y = clean_audio(y)
    target_len = SR * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
    return y.astype(np.float32)

def _extract_mel(y):
    """Branch 1: Log-Mel-Spectrogram → (1, N_MELS, TIME_FRAMES, 1)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, center=False
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    if mel_db.shape[1] < TIME_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, TIME_FRAMES - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :TIME_FRAMES]
    return mel_db[:, :, np.newaxis][np.newaxis, ...].astype(np.float32)  # (1,128,215,1)

def _extract_yamnet_mfcc(y):
    """Branch 2: YAMNet mean+max + MFCCs → (1, 2148)."""
    scores, embeddings, _ = yamnet_model(y)
    emb = embeddings.numpy()
    yamnet_feats = np.concatenate([np.mean(emb, axis=0), np.max(emb, axis=0)])  # 2048

    mfcc     = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40)
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SR)
    zcr      = librosa.feature.zero_crossing_rate(y=y)
    rms      = librosa.feature.rms(y=y)

    trad = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        [np.mean(centroid), np.std(centroid)],
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        [np.mean(zcr), np.std(zcr)],
        [np.mean(rms), np.std(rms)]
    ])
    return np.concatenate([yamnet_feats, trad]).reshape(1, -1).astype(np.float32)

def extract_features(file_path: str):
    """Returns model input(s): list [mel, yamnet] for dual-branch, or single array for legacy."""
    y = _load_uniform(file_path)
    if IS_DUAL:
        return [_extract_mel(y), _extract_yamnet_mfcc(y)]
    else:
        # Legacy single-branch fallback
        scores, embeddings, _ = yamnet_model(y)
        emb = embeddings.numpy()
        yamnet_feats = np.concatenate([np.mean(emb, axis=0), np.max(emb, axis=0)])
        
        mfcc     = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40)
        centroid = librosa.feature.spectral_centroid(y=y, sr=SR)
        contrast = librosa.feature.spectral_contrast(y=y, sr=SR)
        zcr      = librosa.feature.zero_crossing_rate(y=y)
        rms      = librosa.feature.rms(y=y)
        
        chroma   = librosa.feature.chroma_stft(y=y, sr=SR)
        rolloff  = librosa.feature.spectral_rolloff(y=y, sr=SR)
        
        mel      = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=40)
        mel_db   = librosa.power_to_db(mel, ref=np.max)
        
        trad = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            [np.mean(centroid), np.std(centroid)],
            np.mean(contrast, axis=1), np.std(contrast, axis=1),
            [np.mean(zcr), np.std(zcr)],
            [np.mean(rms), np.std(rms)],
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            [np.mean(rolloff), np.std(rolloff)],
            np.mean(mel_db, axis=1), np.std(mel_db, axis=1)
        ])
        feat = np.concatenate([yamnet_feats, trad]).reshape(1, -1).astype(np.float32)
        return feat


# ── Visual Spectrogram Generation & Popup Window Trigger ───────────────────────
def generate_spectrogram_plot(audio_path, prediction_label):
    """Generate waveform + MFCC plot and save as static/temp_plot.png."""
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        y = librosa.util.normalize(y)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        info = CRY_INFO.get(prediction_label, {})
        color = info.get('color', '#22d3ee')
        emoji = info.get('emoji', '👶')
        label = info.get('label', prediction_label.replace('_', ' ').title())
        
        plt.rcParams['text.color'] = '#e2e8f0'
        plt.rcParams['axes.labelcolor'] = '#94a3b8'
        plt.rcParams['xtick.color'] = '#64748b'
        plt.rcParams['ytick.color'] = '#64748b'
        
        fig = plt.figure(figsize=(10, 5.5), facecolor='#090d16')
        fig.suptitle(
            f"CrySense Analysis: {emoji} {label.upper()}",
            color=color,
            fontsize=14,
            fontweight='bold',
            y=0.96
        )
        
        # Plot 1: Waveform
        ax1 = plt.subplot(2, 1, 1, facecolor='#0d1321')
        librosa.display.waveshow(y, sr=sr, ax=ax1, color=color, alpha=0.85)
        ax1.set_title("RAW AUDIO INPUT (AMPLITUDE VS TIME)", color='#94a3b8', fontsize=9, fontweight='bold', pad=5)
        ax1.grid(True, linestyle=':', alpha=0.2, color='#ffffff')
        
        # Plot 2: Spectrogram
        ax2 = plt.subplot(2, 1, 2, facecolor='#0d1321')
        img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax2, cmap='viridis')
        ax2.set_title("EXTRACTED MFCC SPECTROGRAM (COEFFICIENTS VS TIME)", color='#94a3b8', fontsize=9, fontweight='bold', pad=5)
        
        # Colorbar
        cbar = fig.colorbar(img, ax=ax2, format='%+2.0f')
        cbar.ax.yaxis.set_tick_params(color='#64748b')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#94a3b8', fontsize=8)
        
        plt.tight_layout(rect=[0, 0.01, 1, 0.93])
        
        os.makedirs('static', exist_ok=True)
        plot_path = os.path.join('static', 'temp_plot.png')
        plt.savefig(plot_path, dpi=120, facecolor='#090d16')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[ERR] Failed to generate plot: {e}")
        return False

def trigger_visual_popup(audio_path, prediction_label):
    """Save raw audio to static folder and launch visualize_prediction.py as subprocess."""
    try:
        os.makedirs('static', exist_ok=True)
        persistent_path = os.path.join('static', 'temp_analysis.wav')
        shutil.copy(audio_path, persistent_path)
        
        # Run python script asynchronously, forwarding stdout/stderr to server logs
        subprocess.Popen(
            [sys.executable, 'visualize_prediction.py', persistent_path, prediction_label],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print(f"[OK] Visualizer subprocess launched for label: {prediction_label}")
    except Exception as e:
        print(f"[ERR] Failed to launch visualizer subprocess: {e}")


# ── Prediction ─────────────────────────────────────────────────────────────────
def run_prediction(file_path: str) -> dict:
    if model is None or label_enc is None:
        return {'error': 'Model not loaded. Please run train_model.py first.'}

    try:
        feat = extract_features(file_path)
        probs = model.predict(feat, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_cls = label_enc.classes_[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        if confidence < 60:
            pred_cls = "uncertain"

        all_probs = {
            label_enc.classes_[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(probs))
        }

        # Generate scientific plot
        generate_spectrogram_plot(file_path, pred_cls)
        # Trigger desktop visualization popup
        trigger_visual_popup(file_path, pred_cls)

        info = CRY_INFO.get(pred_cls, {})

        # Infer secondary emotion
        secondary = infer_secondary_emotion(pred_cls, confidence, all_probs)

        result = {
            'prediction':         pred_cls,
            'label':              info.get('label', pred_cls.replace('_', ' ').title()),
            'confidence':         round(confidence, 2),
            'emoji':              info.get('emoji', '👶'),
            'color':              info.get('color', '#6c63ff'),
            'advice':             info.get('advice', ''),
            'advice_hi':          info.get('advice_hi', ''),
            'quick_tips':         info.get('quick_tips', []),
            'severity':           info.get('severity', 'low'),
            'doctor_warning':     info.get('doctor_warning', ''),
            'duration_tip':       info.get('duration_tip', ''),
            'secondary_emotions': info.get('secondary_emotions', []),
            'inferred_emotion':   secondary,
            'all_probs':          all_probs,
            'plot_url':           '/static/temp_plot.png',
            'status':             'success'
        }
        return result

    except Exception as e:
        return {'error': str(e), 'status': 'error'}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided', 'status': 'error'}), 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename', 'status': 'error'}), 400

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)
    try:
        result = run_prediction(tmp_path)
    finally:
        os.unlink(tmp_path)
    return jsonify(result)


@app.route('/model-info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'status': 'not_trained', 'message': 'Model not found. Run train_model.py.'})
    classes  = list(label_enc.classes_) if label_enc else []
    best_acc = None
    if training_history:
        best_acc = round(max(training_history.get('val_accuracy', [0])) * 100, 2)
    return jsonify({
        'status':        'ready',
        'classes':       classes,
        'best_val_acc':  best_acc,
        'model_path':    MODEL_PATH,
        'total_params':  model.count_params() if model else 0
    })


@app.route('/history', methods=['GET'])
def history():
    if not training_history:
        return jsonify({'error': 'No training history found'}), 404
    return jsonify(training_history)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':          'online',
        'model_loaded':    model is not None,
        'encoder_loaded':  label_enc is not None
    })


# ── Emotion info endpoint ───────────────────────────────────────────────────────
@app.route('/emotions', methods=['GET'])
def emotions():
    """Return all emotion metadata for the frontend."""
    return jsonify({
        'primary':   CRY_INFO,
        'inferred':  INFERRED_EMOTIONS
    })


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  CrySense Backend  --  Baby Cry Detection API")
    print("="*55)
    load_resources()
    print("\n  Running at -> http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
