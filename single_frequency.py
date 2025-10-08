import numpy as np
import matplotlib.pyplot as plt
import time

from mydaq import MyDAQ


frequency = 500  # Hz

now = time.strftime("%Y%m%d%H%M%S")


daq = MyDAQ(44100, 'myDAQ4')

_, sine = MyDAQ.generateWaveform(
    'sine',
    samplerate=daq.samplerate,
    frequency=frequency,
    amplitude=0.25,
    duration=5,
)

data = daq.readWrite(sine, write_channel='ao0', read_channel='ai1')
np.save(f'data/single_frequency_{frequency}Hz_{now}.npy', data)

fft = np.fft.fft(data)
fft_freq = np.fft.fftfreq(len(data), 1/daq.samplerate)
dB = 20*np.log10(np.abs(fft))

fig, ax = plt.subplots()

idx = MyDAQ.find_nearest_idx(fft_freq, frequency)
ax.vlines(
    fft_freq[idx],
    ymin=np.min(dB),
    ymax=np.max(dB),
    colors='r',
    linestyles='dashed',
    alpha=0.5,
    linewidth=1,
)

ax.scatter(
    fft_freq[:len(fft)//2],
    dB[:len(fft)//2],
    s=1,
    marker='.',
    color='k',
)

ax.set(
    xscale='log',
    xlabel='Frequency (Hz)',
    ylabel='Magnitude (dB)',
    title=f'Single Frequency {frequency} Hz',
)

plt.savefig(f'figures/single_frequency_{frequency}Hz_{now}.png', dpi=300)
plt.savefig(f'figures/single_frequency_{frequency}Hz_{now}.pdf', dpi=300)
