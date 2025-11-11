import numpy as np 
import matplotlib.pyplot as plt

from mydaq import MyDAQ

file = r'data\20251107135340_buurman_nils.npy'

data = np.load(file)

daq = MyDAQ(44100, 'myDAQ1')

fft = np.fft.fft(data)
fft_freq = np.fft.fftfreq(len(data), 1/daq.samplerate)
dB = 20*np.log10(np.abs(fft))

fig, ax = plt.subplots(1, 2, figsize=(16, 10), layout='tight')

ax[0].scatter(
    fft_freq[:len(fft)//2],
    dB[:len(fft)//2],
    s=4,
    marker='.',
    color='k',
)

ax[0].set_xscale('log')
ax[0].set_xlabel('Frequency (Hz)', fontsize=24)
ax[0].set_ylabel('Magnitude (dB)', fontsize=24)
ax[0].set_title(f'Fourier plot of {file}', fontsize=28)
ax[0].tick_params(labelsize = 16)
ax[0].grid()

ax[1].scatter(
    np.arange(len(data)),
    data,
    s=4,
    marker='.',
    color='k',
)

ax[1].set_xscale('log')
ax[1].set_xlabel('Frequency (Hz)', fontsize=24)
ax[1].set_ylabel('Magnitude (dB)', fontsize=24)
ax[1].set_title(f'Fourier plot of {file}', fontsize=28)
ax[1].tick_params(labelsize = 16)
ax[1].grid()

plt.savefig(file.replace('.npy', '.png').replace('data', 'figures'), dpi=300)
plt.savefig(file.replace('.npy', '.pdf').replace('data', 'figures'), dpi=300)