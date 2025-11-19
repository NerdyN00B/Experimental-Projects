# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 17:33:18 2025

@author: lotte
"""

from pathlib import Path
import torch
import librosa
from torchaudio.pipelines import SQUIM_OBJECTIVE
import numpy as np
import matplotlib.pyplot as plt

def compute_SII(STOI):
    return 100 / (1 + np.exp(-13.19 * STOI + 6.52))
SII_list = []

objective_model = SQUIM_OBJECTIVE.get_model()
folder = Path("C:/Users/lotte/_uni/jaar 4/EP/data/tests")

for filepath in folder.glob("*.wav"):   # loop over all .wav files in deze map
    print(f"\nProcessing: {filepath.name}")

    waveform, sr = librosa.load(filepath, sr=16000)

    # Convert to tensor with shape (1, time)
    waveform = torch.tensor(waveform).float().unsqueeze(0)

    stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(waveform)
    stoi_hyp = stoi_hyp.tolist()
    SII_list.append(compute_SII(stoi_hyp[0]))

    #print(f"STOI: {stoi_hyp[0]}")
    print(f"SII gebaseerd op STOI: {compute_SII(stoi_hyp[0])}")
    #print(f"PESQ: {pesq_hyp[0]}")
    #print(f"SI-SDR: {si_sdr_hyp[0]}")


SII_arr = np.array(SII_list).reshape(5,3)

# Plot heatmap
plt.imshow(SII_arr)
plt.colorbar(label="SII")

plt.title("SII per Location")
plt.tight_layout()
plt.show()