import numpy as np
import matplotlib.pyplot as plt
import time

from dualdaq import DualDaq

amplitude = 0.5
duration = 1.1
measurements = 3

freqs = np.logspace(np.log10(400), np.log10(14000), 250, dtype=int)

daq = DualDaq(44100)

for i in range(10):
    print(10-i)
    time.sleep(1)

full_measurement = []
for i, freq in enumerate(freqs):
    print(f'{freq}, {i+1} / {len(freqs)}')
    sine = daq.gensine(freq, duration, amplitude)
    single_freq_measurement = []
    for _ in range(measurements):
        data = daq.readwritedual(
            sine,
            'myDAQ2/ao0',
            'myDAQ1/ai0',
        )

        single_freq_measurement.append(data)
    full_measurement.append(np.asarray(single_freq_measurement))

full_measurement = np.asarray(full_measurement)

full_transfer = np.fft.fft(
    full_measurement[:, :, :int(-0.1*daq.samplerate)],
)

file = f'data/{time.strftime("%Y%m%d%H%M%S")}_errored_transfer.npy'

np.save(
    file,
    full_transfer,
)


found_transfer_mean = []
found_transfer_std = []
found_transfer = []
for i, freq in enumerate(freqs):
    fftfreq = np.fft.fftfreq(
        int((duration - 0.1) * daq.samplerate),
        1/daq.samplerate
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
    yerr=found_transfer_std*10,
    fmt='k.',
    markersize=15,
    capsize=10,
    label=r'Average measured transfer function $\pm 10\sigma$',
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
    file.replace('.npy', '.png').replace('data', 'figures'),
    dpi=300
)