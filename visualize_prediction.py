import sys
import os
import numpy as np
import librosa
import librosa.display
import matplotlib
# Force TkAgg backend for visual window popup on desktop
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Primary cry types configuration for visual labels & colors
CRY_STYLES = {
    'belly_pain': {'emoji': '🤢', 'color': '#ef4444', 'label': 'Belly Pain / Gas'},
    'burping': {'emoji': '💨', 'color': '#f97316', 'label': 'Needs to Burp'},
    'discomfort': {'emoji': '😣', 'color': '#eab308', 'label': 'Discomfort / Irritation'},
    'hungry': {'emoji': '🍼', 'color': '#22c55e', 'label': 'Hungry / Needs Feeding'},
    'tired': {'emoji': '😴', 'color': '#6c63ff', 'label': 'Tired / Sleepy'},
    'cold_hot': {'emoji': '🌡️', 'color': '#06b6d4', 'label': 'Too Cold / Too Hot'},
    'uncertain': {'emoji': '👶', 'color': '#94a3b8', 'label': 'Uncertain Cry'}
}

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_prediction.py <audio_path> <prediction_label>")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    pred_label = sys.argv[2]
    
    if not os.path.exists(audio_path):
        print(f"Error: File not found {audio_path}")
        sys.exit(1)
        
    # Get styling properties for the predicted class
    style = CRY_STYLES.get(pred_label, {'emoji': '👶', 'color': '#00f0ff', 'label': pred_label.replace('_', ' ').title()})
    
    print(f"Generating visual plots for: {style['label']}...")
    
    try:
        # Load audio at 16kHz
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Normalize
        y = librosa.util.normalize(y)
        
        # Calculate MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Set up modern dark plotting theme
        plt.rcParams['text.color'] = '#e2e8f0'
        plt.rcParams['axes.labelcolor'] = '#94a3b8'
        plt.rcParams['xtick.color'] = '#64748b'
        plt.rcParams['ytick.color'] = '#64748b'
        plt.rcParams['font.family'] = 'sans-serif'
        
        fig = plt.figure(figsize=(10, 6.5), facecolor='#090d16')
        
        # Super title showing predicted category with emoji
        fig.suptitle(
            f"CrySense Analysis: {style['emoji']} {style['label'].upper()}",
            color=style['color'],
            fontsize=16,
            fontweight='bold',
            y=0.96
        )
        
        # Plot 1: Raw Audio Waveform
        ax1 = plt.subplot(2, 1, 1, facecolor='#0d1321')
        librosa.display.waveshow(y, sr=sr, ax=ax1, color=style['color'], alpha=0.85)
        ax1.set_title("RAW AUDIO INPUT (AMPLITUDE VS TIME)", color='#94a3b8', fontsize=10, fontweight='bold', pad=8)
        ax1.set_xlabel("Time (seconds)", fontsize=9)
        ax1.set_ylabel("Amplitude", fontsize=9)
        ax1.grid(True, linestyle=':', alpha=0.2, color='#ffffff')
        ax1.spines['bottom'].set_color('#1e293b')
        ax1.spines['top'].set_color('#1e293b')
        ax1.spines['left'].set_color('#1e293b')
        ax1.spines['right'].set_color('#1e293b')
        
        # Plot 2: MFCC Spectrogram
        ax2 = plt.subplot(2, 1, 2, facecolor='#0d1321')
        img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax2, cmap='viridis')
        ax2.set_title("EXTRACTED MFCC SPECTROGRAM (COEFFICIENTS VS TIME)", color='#94a3b8', fontsize=10, fontweight='bold', pad=8)
        ax2.set_xlabel("Time (seconds)", fontsize=9)
        ax2.set_ylabel("MFCC Coefficients", fontsize=9)
        ax2.spines['bottom'].set_color('#1e293b')
        ax2.spines['top'].set_color('#1e293b')
        ax2.spines['left'].set_color('#1e293b')
        ax2.spines['right'].set_color('#1e293b')
        
        # Colorbar
        cbar = fig.colorbar(img, ax=ax2, format='%+2.0f')
        cbar.ax.yaxis.set_tick_params(color='#64748b')
        cbar.outline.set_edgecolor('#1e293b')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#94a3b8', fontsize=8)
        cbar.set_label("dB Power", color='#94a3b8', fontsize=9)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        
        # Try to bring window to front
        try:
            mngr = plt.get_current_fig_manager()
            if hasattr(mngr, 'window'):
                mngr.window.attributes('-topmost', 1)
                mngr.window.attributes('-topmost', 0)
                mngr.window.focus_force()
        except Exception as ex:
            print(f"Focus window tip: {ex}")
            
        plt.show()
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
