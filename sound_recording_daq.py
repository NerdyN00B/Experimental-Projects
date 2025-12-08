import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io.wavfile import write
from mydaq import MyDAQ

now  = time.strftime("%Y%m%d%H%M%S")

daq = MyDAQ(44100, 'myDAQ1')

file = f'data/{now}_RandomStilte.npy'
print("Recording...")

data = daq.read(10, channel='ai0')

np.save(file, data)

fig, ax = plt.subplots(figsize=(16, 10), layout='tight')

fft =  np.fft.fft(data)
fft_freq = np.fft.fftfreq(len(data), 1/daq.samplerate)

dB = 20*np.log10(np.abs(fft))

ax.scatter(
    fft_freq[:len(fft)//2],
    dB[:len(fft)//2],
    s=10,
    marker='.',
    color='k',
)

ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)', fontsize=24)
ax.set_ylabel('Magnitude (dB)', fontsize=24)
ax.set_title(f'10 second recording of some sort', fontsize=28)
ax.tick_params(labelsize = 16)
ax.grid()

fig.savefig(file.replace('.npy', '_spectrum.png').replace('data', 'figures'), dpi=300)

data -= np.mean(data)  # Remove DC offset
data = data / np.max(np.abs(data))  # Normalize to -1 to 1

amplitude = np.iinfo(np.int16).max
data = (data * amplitude).astype(np.int16)

write(
    file.replace('.npy', '.wav').replace('data', 'audio'),
    daq.samplerate,
    data
)
