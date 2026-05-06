import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def generate_spectrograms():
    # Paths
    real_data_path = r"D:\Antigravity\root\Datasets\RealData\Peugeot307\Peugeot307_103.wav"
    sim_data_path = r"D:\Antigravity\root\Datasets\SimulatedData\Peugeot307\Peugeot307_103.0.wav"
    clean_sim_path = r"D:\Antigravity\root\DopplerSim\static\batch_outputs\CleanSimulatedData\Peugeot307\Peugeot307_103.wav"
    
    output_dir = r"D:\Antigravity\root\DopplerSim\temp"
    os.makedirs(output_dir, exist_ok=True)

    paths = [real_data_path, sim_data_path, clean_sim_path]
    titles = ["RealData", "SimulatedData", "CleanSimulatedData"]
    main_title = "Peugeot307 - 103km/h"
    out_path = os.path.join(output_dir, "Peugeot307_103_spectrograms.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(main_title, fontsize=16)

    n_fft = 4096
    hop_length = 512
    win_length = 4096
    window = "hann"
    max_y_freq = 1250.0

    for i, ax in enumerate(axes):
        path = paths[i]
        title = titles[i]
        if not os.path.exists(path):
            ax.set_title(f"{title}\n(File not found)")
            print(f"File not found: {path}")
            continue
        
        y, sr = librosa.load(path, sr=None)
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        s_power = np.abs(stft) ** 2
        d_db = librosa.power_to_db(s_power, ref=np.max)
        vmax = float(np.max(d_db))
        vmin = vmax - 80.0
        
        librosa.display.specshow(
            d_db,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='hz',
            ax=ax,
            cmap='magma',
            rasterized=True,
            vmin=vmin,
            vmax=vmax
        )
        ax.set_ylim(0, max_y_freq)
        ax.set_yticks(np.linspace(0, max_y_freq, 6))
        ax.set_title(title)
        if i == 0:
            ax.set_ylabel('Frequency (Hz)')
        else:
            ax.set_ylabel('')
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved spectrograms to {out_path}")

if __name__ == '__main__':
    generate_spectrograms()
