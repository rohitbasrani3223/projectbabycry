"""
CrySense - Flask Backend API
==============================
Serves predictions from the trained CNN model.
Endpoints:
  POST /predict        - Upload .wav file → get cry type prediction
  POST /predict-live   - Record from mic (5 sec) → predict
  GET  /model-info     - Model accuracy, classes, training stats
  GET  /history        - Training history JSON
"""

import os
import io
import json
import pickle
import tempfile
import numpy as np
import librosa
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# ── Load model on startup ──────────────────────────────────────────────────────
MODEL_PATH   = 'crysense_model.h5'
ENCODER_PATH = 'label_encoder.pkl'
HISTORY_PATH = 'training_history.json'

app = Flask(__name__, static_folder='static')
CORS(app)

model       = None
label_enc   = None
training_history = {}

# Emoji + advice map for each cry type
CRY_INFO = {
    'belly_pain': {
        'emoji': '🤢',
        'color': '#ef4444',
        'advice': 'Baby may have gas or stomach pain. Try gentle tummy massage in clockwise circles, bicycle leg movements, or burping the baby.',
        'severity': 'medium'
    },
    'burping': {
        'emoji': '💨',
        'color': '#f97316',
        'advice': 'Baby needs to burp. Hold baby upright, gently pat or rub the back until they burp.',
        'severity': 'low'
    },
    'discomfort': {
        'emoji': '😣',
        'color': '#eab308',
        'advice': 'Baby is uncomfortable. Check diaper, clothing, temperature, or if anything is irritating the skin.',
        'severity': 'low'
    },
    'hungry': {
        'emoji': '🍼',
        'color': '#22c55e',
        'advice': 'Baby is hungry! Time to feed. Offer breast or bottle. Look for rooting reflex signs.',
        'severity': 'medium'
    },
    'tired': {
        'emoji': '😴',
        'color': '#6c63ff',
        'advice': 'Baby is tired and needs sleep. Dim the lights, reduce noise, try rocking or swaddling.',
        'severity': 'low'
    }
}


def load_resources():
    global model, label_enc, training_history
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[OK] Model loaded from {MODEL_PATH}")
        else:
            print(f"[WARN] Model not found at {MODEL_PATH} -- run train_model.py first")
    except Exception as e:
        print(f"[ERR] Error loading model: {e}")

    try:
        if os.path.exists(ENCODER_PATH):
            with open(ENCODER_PATH, 'rb') as f:
                label_enc = pickle.load(f)
            print(f"[OK] Label encoder loaded")
    except Exception as e:
        print(f"[ERR] Error loading encoder: {e}")

    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r') as f:
                training_history = json.load(f)
    except:
        pass


def extract_mfcc(file_path: str) -> np.ndarray:
    N_MFCC  = 40
    MAX_LEN = 128
    SR      = 22050
    y, sr = librosa.load(file_path, sr=SR, duration=5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return mfcc[np.newaxis, ..., np.newaxis]   # (1, 40, 128, 1)


def run_prediction(file_path: str) -> dict:
    if model is None or label_enc is None:
        return {'error': 'Model not loaded. Please run train_model.py first.'}

    try:
        features = extract_mfcc(file_path)
        probs    = model.predict(features, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_cls = label_enc.classes_[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        all_probs = {
            label_enc.classes_[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(probs))
        }

        info = CRY_INFO.get(pred_cls, {})
        return {
            'prediction':  pred_cls,
            'confidence':  round(confidence, 2),
            'emoji':       info.get('emoji', '👶'),
            'color':       info.get('color', '#6c63ff'),
            'advice':      info.get('advice', ''),
            'severity':    info.get('severity', 'low'),
            'all_probs':   all_probs,
            'status':      'success'
        }
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
        return jsonify({
            'status': 'not_trained',
            'message': 'Model not found. Run train_model.py to train.'
        })

    classes = list(label_enc.classes_) if label_enc else []
    best_acc = None
    if training_history:
        best_acc = round(max(training_history.get('val_accuracy', [0])) * 100, 2)

    return jsonify({
        'status':      'ready',
        'classes':     classes,
        'best_val_acc': best_acc,
        'model_path':  MODEL_PATH,
        'total_params': model.count_params() if model else 0
    })


@app.route('/history', methods=['GET'])
def history():
    if not training_history:
        return jsonify({'error': 'No training history found'}), 404
    return jsonify(training_history)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':        'online',
        'model_loaded':  model is not None,
        'encoder_loaded': label_enc is not None
    })


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  CrySense Backend  --  Baby Cry Detection API")
    print("="*55)
    load_resources()
    print("\n  Running at -> http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
