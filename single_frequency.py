import numpy as np
import matplotlib.pyplot as plt
import time

from mydaq import MyDAQ


frequency = 500  # Hz
amplitude = .3  # V

now = time.strftime("%Y%m%d%H%M%S")


daq = MyDAQ(44100, 'myDAQ4')

_, sine = MyDAQ.generateWaveform(
    'sine',
    samplerate=daq.samplerate,
    frequency=frequency,
    amplitude=amplitude,
    duration=5,
)

data = daq.readWrite(sine, write_channel='ao0', read_channel='ai1')
np.save(f'data/{now}_single_frequency_{frequency}Hz.npy', data)

fft = np.fft.fft(data)
fft_freq = np.fft.fftfreq(len(data), 1/daq.samplerate)
dB = 20*np.log10(np.abs(fft))

fig, ax = plt.subplots(figsize=(16, 10), layout='tight')

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

ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)', fontsize=16)
ax.set_ylabel('Magnitude (dB)', fontsize=16)
ax.set_title(f'Single Frequency Response at {frequency} Hz', fontsize=20)
ax.grid()

plt.savefig(f'figures/{now}_single_frequency_{frequency}Hz.png', dpi=300)
plt.savefig(f'figures/{now}_single_frequency_{frequency}Hz.pdf', dpi=300)
