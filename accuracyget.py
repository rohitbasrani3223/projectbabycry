import os, json, pickle
print('=== CrySense Project Status ===')
files = ['crysense_model.pkl', 'label_encoder.pkl', 'feature_scaler.pkl', 'training_history.json']
for f in files:
    exists = os.path.exists(f)
    if exists:
        import time
        mtime = time.ctime(os.path.getmtime(f))
        print(f'  [OK] {f}  ({mtime})')
    else:
        print(f'  [X]  {f}  (NOT FOUND)')

if os.path.exists('training_history.json'):
    h = json.load(open('training_history.json'))
    if 'val_accuracy' in h:
        va = h['val_accuracy']
        print(f'\n  Best Val Accuracy: {max(va)*100:.2f}%')
        print(f'  Epochs completed : {len(va)}')
