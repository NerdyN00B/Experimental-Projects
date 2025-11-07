import numpy as np
import matplotlib.pyplot as plt
import time

from dualdaq import DualDaq

freq = 500

daq = DualDaq(44100)

duration = 1.1

voltages = np.logspace(-1, 1, 12)

transfer_functions = []
for voltage in voltages:
    sine = daq.gensine(freq, duration, voltage)
    read_data = daq.readwritedual(
        sine,
        'myDAQ1/ao0',
        'myDAQ2/ai0',
    )
    transfer_functions.append(read_data / sine)
    fft = np.fft.fft(read_data[:-0.1*daq.samplerate])
    transfer_functions.append(fft)

now = time.strftime('%Y%m%d%H%M%S')
fftfreq = np.fft.fftfreq(44100, 1/44100)
idx = np.argmin(np.abs(fftfreq - 500))
np.save(f'{now}_min_voltage_test.npy', np.asarray(transfer_functions))

fig, ax = plt.subplots(3, 4)
for transfer in transfer_functions:
    db = 20 * np.log10(np.abs(transfer))
    ax.scatter(
        fftfreq[:len(transfer)//2],
        db[:len(transfer)//2],
        s=10,
        marker='.',
        color='k',
    )
    ax.vlines(
        freq,
        np.min(db),
        np.max(db),
        colors='r',
        linestyles='dashed',
        alpha=0.5,
        linewidth=1,
    )
    ax.set(
        xscale='log',
        xlabel='Frequency (Hz)',
        ylabel='Magnitude (dB)',
        title='Transfer Function Minimum Voltage Test',
    )
    ax.xlim(freq-50, freq+50)

fig.savefig(f'figures/{now}_min_voltage_test.png', dpi=300)