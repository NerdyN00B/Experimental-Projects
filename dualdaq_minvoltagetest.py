import numpy as np
import matplotlib.pyplot as plt
import time

from dualdaq import DualDaq

freq = 10000

daq = DualDaq(44100)

duration = 1.1

voltages = np.logspace(-1, 0, 12)

transfer_functions = []
for i, voltage in enumerate(voltages):
    print(f'{voltage}, ({i+1} / {len(voltages)})')
    sine = daq.gensine(freq, duration, voltage)
    read_data = daq.readwritedual(
        sine,
        'myDAQ2/ao0',
        'myDAQ1/ai0',
    )
    fft = np.fft.fft(read_data[:int(-0.1*daq.samplerate)])
    transfer_functions.append(fft)

now = time.strftime('%Y%m%d%H%M%S')
fftfreq = np.fft.fftfreq(44100, 1/44100)
idx = np.argmin(np.abs(fftfreq - 500))
np.save(f'data/{now}_min_voltage_test_{freq}Hz.npy', np.asarray(transfer_functions))

fig, ax = plt.subplots(3, 4, layout='tight', figsize=(16, 10))

for i, transfer in enumerate(transfer_functions):
    x = i%4
    y = i//4
    db = 20 * np.log10(np.abs(transfer))
    ax[y, x].scatter(
        fftfreq[:len(transfer)//2],
        db[:len(transfer)//2],
        s=10,
        marker='.',
        color='k',
    )
    ax[y, x].vlines(
        freq,
        np.min(db),
        np.max(db),
        colors='r',
        linestyles='dashed',
        alpha=0.5,
        linewidth=1,
    )
    ax[y, x].set(
        xlabel='Frequency (Hz)',
        ylabel='Magnitude (dB)',
        title=f'{voltages[i]:.2f} V',
    )
    ax[y, x].set_xlim(freq-50, freq+50)

fig.suptitle("Test for minimum voltage")

fig.savefig(f'figures/{now}_min_voltage_test_{freq}Hz.png', dpi=300)