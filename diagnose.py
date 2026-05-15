import librosa, numpy as np
from pathlib import Path

base = Path('aug-dataset1')
classes = ['belly_pain','burping','cold_hot','discomfort','hungry','tired']
print("=== Dataset Diagnosis ===")
for cls in classes:
    d = base / cls
    files = list(d.glob('*.wav'))
    durs = []
    for f in files[:5]:
        try:
            y, sr = librosa.load(str(f), sr=22050)
            durs.append(len(y)/sr)
        except:
            pass
    avg = sum(durs)/len(durs) if durs else 0
    print(f"  {cls:<15} files={len(files):>4}  avg_dur={avg:.2f}s")
