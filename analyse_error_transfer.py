import numpy as np
import matplotlib.pyplot as plt

file = r'data\20251114155231_errored_transfer.npy'

full_transfer = np.load(file)
freqs = np.logspace(np.log10(400), np.log10(14000), 200, dtype=int)

duration = 1.1
samplerate = 44100

found_transfer_mean = []
found_transfer_std = []
found_transfer = []
for i, freq in enumerate(freqs):
    fftfreq = np.fft.fftfreq(
        int((duration - 0.1) * samplerate),
        1/samplerate
    )
    
    idx = np.argmin(np.abs(fftfreq - freq))
    true_idx = np.argmax(np.abs(full_transfer[i, 0, idx-50:idx+50]))
    true_idx += idx - 50
    
    power = np.trapezoid(
        full_transfer[i, :, true_idx-5:true_idx+5],
        fftfreq[true_idx-5:true_idx+5],
        axis=-1,
    )
    found_transfer.append(power)
    db = 20 * np.log10(np.abs(power))
    
    found_transfer_mean.append(np.mean(db))
    found_transfer_std.append(np.std(db))

fig, ax = plt.subplots(figsize=(16, 10), layout='tight')

found_transfer_mean = np.asarray(found_transfer_mean)
found_transfer_std = np.asarray(found_transfer_std)
found_transfer = np.asarray(found_transfer)


ax.errorbar(
    freqs,
    found_transfer_mean,
    yerr=found_transfer_std,
    fmt='k.',
    markersize=15,
    capsize=10,
    label=r'Average measured transfer function $\pm \sigma$',
    linestyle=':'
)

# for i in range(3):
#     ax.scatter(
#         freqs,
#         20 * np.log10(np.abs(found_transfer[:, i])),
#         s=200,
#     )

ax.legend(fontsize=24)
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)', fontsize=24)
ax.set_ylabel('Magnitude (dB)', fontsize=24)
ax.set_title(
    'Transfer function measurement without reference signal',
    fontsize=28
)

ax.tick_params(labelsize = 16)

fig.savefig(
    file.replace('.npy', '_reprocessed.png').replace('data', 'figures'),
    dpi=300
)
