import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap


def find_freqs(data):
    n = data.shape[0]
    return np.logspace(np.log10(400), np.log10(14000), n, dtype=int)


def determine_transfer(data, freqs):
    found_transfer_mean = []
    found_transfer_std = []
    for i, freq in enumerate(freqs):
        fftfreq = np.fft.fftfreq(
            int((duration - 0.1) * samplerate),
            1/samplerate
        )

        idx = np.argmin(np.abs(fftfreq - freq))
        true_idx = np.argmax(np.abs(data[i, 0, idx-50:idx+50]))
        true_idx += idx - 50

        # Using depricated trapz function instead of trapezoid as I'm working
        # on an older numpy version in this environment.
        power = np.trapz(
            data[i, :, true_idx-5:true_idx+5] ** 2,
            fftfreq[true_idx-5:true_idx+5],
            axis=-1,
        )
        power = np.sqrt(power)
        db = 20 * np.log10(np.abs(power))

        found_transfer_mean.append(np.mean(db))
        found_transfer_std.append(np.std(db))
    
    return np.asarray(found_transfer_mean), np.asarray(found_transfer_std)


dataset = {
    "5 cm": r"data\20251121131436_errored_transfer.npy",
    "4 cm": r"data\20251126112534_errored_transfer_4cm.npy",
    "3 cm": r"data\20251126114813_errored_transfer_3cm.npy",
    "2 cm": r"data\20251126120628_errored_transfer_2cm.npy",
    "1 cm": r"data\20251126121833_errored_transfer_1cm.npy",
}

duration = 1.1
samplerate = 44100

fig, ax = plt.subplots(figsize=(8, 4), layout='tight')


for label, filepath in dataset.items():
    data = np.load(filepath)
    freqs = find_freqs(data)
    mean, std = determine_transfer(data, freqs)
    
    cmap = get_cmap('viridis')
    radius = int(label.split()[0])
    color = cmap(radius / 5)
    
    ax.errorbar(
        freqs,
        mean,
        yerr=std,
        fmt='.',
        markersize=8,
        capsize=3,
        color=color,
        linestyle=':',
        label=f"radius = {label}",
    )

ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Loudness (dB)')
ax.set_title('Transfer Function Measurements at Various Drum Sizes')
ax.legend()
ax.grid()

# plt.show()

filename = r'figures/final_analysis/radius_range_transfer_functions.png'
fig.savefig(filename, dpi=300)
fig.savefig(filename.replace('.png', '.pdf'), dpi=300)

fig, ax = plt.subplots(figsize=(16, 10), layout='tight')

for label, filepath in dataset.items():
    data = np.load(filepath)
    freqs = find_freqs(data)
    mean, std = determine_transfer(data, freqs)
    
    cmap = get_cmap('viridis')
    radius = int(label.split()[0])
    color = cmap(radius / 5)
    
    ax.errorbar(
        freqs,
        mean,
        yerr=std,
        fmt='.',
        markersize=15,
        capsize=5,
        color=color,
        linestyle=':',
        label=f"radius = {label}",
    )

ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)', fontsize=24)
ax.set_ylabel('Loudness (dB)', fontsize=24)
ax.set_title('Transfer Function Measurements at Various Drum Sizes',
             fontsize=30)
ax.legend(fontsize=14)
ax.grid()
ax.tick_params(labelsize = 20)


fig.savefig(
    filename.replace('transfer_functions', 
                     'transfer_functions_presentation'), 
    dpi=300
)