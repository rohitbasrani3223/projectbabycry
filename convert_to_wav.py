from pydub import AudioSegment
from pathlib import Path

DATASET_DIR = "aug-dataset1"

supported = [".mp3", ".ogg", ".mpeg", ".3gp", ".m4a"]

base = Path(DATASET_DIR)

count = 0

for file in base.rglob("*"):

    if file.suffix.lower() in supported:

        try:

            audio = AudioSegment.from_file(file)

            wav_path = file.with_suffix(".wav")

            audio.export(wav_path, format="wav")

            print(f"Converted: {file.name}")

            count += 1

        except Exception as e:

            print(f"Failed: {file} -> {e}")

print(f"\nDone. Converted {count} files.")