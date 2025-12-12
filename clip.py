import numpy as np
import matplotlib.pyplot as plt
import time

from mydaq import MyDAQ

now  = time.strftime("%Y%m%d%H%M%S")

daq = MyDAQ(44100, 'myDAQ1')

file = r'data\20251212134103_20m.npy'

data = np.load(file)

fig, ax = plt.subplots(figsize=(16, 10), layout='tight')

fft =  np.fft.fft(data)
fft_freq = np.fft.fftfreq(len(data), 1/daq.samplerate)

# Clipping away everything above 1500 Hz
Clipfreq = 6000
smallclip = 0

fft[fft_freq > Clipfreq] = 0 + 0j
fft[fft_freq < -Clipfreq] = 0 + 0j

fft[fft_freq < smallclip] = 0 + 0j

data = abs(np.fft.ifft(fft))
fft = np.fft.fft(data)

np.save(file.replace('.npy', F'_clipped_{Clipfreq}Hz.npy'), data)

# dB = 20*np.log10(np.abs(fft))

ax.scatter(
    fft_freq[:len(fft)//2],
    abs(fft)[:len(fft)//2],
    s=10,
    marker='.',
    color='k',
)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Frequency (Hz)', fontsize=24)
ax.set_ylabel('Magnitude (dB)', fontsize=24)
ax.set_title(f'10 second recording of some sort', fontsize=28)
ax.tick_params(labelsize = 16)
ax.grid()

fig.savefig(file.replace('.npy', '_clipped_spectrum.png').replace('data', 'figures'), dpi=300)
