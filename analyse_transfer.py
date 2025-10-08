import numpy as np
import matplotlib.pyplot as plt

file = r'data\transfer_measurement_20251008110608.npy'
now = file.strip('.npy').split('_')[-1]

data = np.load(file)
frequencies = [300, 500, 1000, 2000, 5000, 10000, 15000]
transfers = data[1]

for i, freq in enumerate(frequencies):
    fig, ax = plt.subplots()
    fft = np.fft.fft(transfers[i])  # First repetition
    freqs = np.fft.fftfreq(len(fft), 1/44100)
    dB = 20*np.log10(np.abs(fft))
    ax.vlines(
        freq,
        np.min(dB),
        np.max(dB),
        colors='r',
        linestyles='dashed',
        alpha=0.5,
        linewidth=1,
    )
    ax.scatter(
        freqs[:len(fft)//2],
        dB[:len(fft)//2],
        s=1,
        marker='.',
        color='k',
    )
    ax.set(
        xscale='log',
        xlabel='Frequency (Hz)',
        ylabel='Magnitude (dB)',
        title=f'Transfer {frequencies[i]} Hz',
    )
    fig.savefig(f'figures/transfer_{frequencies[i]}Hz_{now}.png', dpi=300)
    fig.savefig(f'figures/transfer_{frequencies[i]}Hz_{now}.pdf', dpi=300)