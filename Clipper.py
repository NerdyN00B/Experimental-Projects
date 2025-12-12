import numpy as np
import time
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

filename = "data/20251212123331_20mDicteeSam.npy"
now  = time.strftime("%Y%m%d%H%M%S")

data = np.load(filename)

fft = np.fft.fft(data)
fftfreq = np.fft.fftfreq(len(data), 1/44100)

clipfreq = 100
fft[(fftfreq > clipfreq) | (fftfreq < -clipfreq)] = 0

clipped = np.fft.ifft(fft)

new_filename = filename[:-4] + "clipped" + filename[-4:-1]

np.save(new_filename, clipped)

clipped -= np.mean(data)  # Remove DC offset
clipped = data / np.max(np.abs(data))  # Normalize to -1 to 1
amplitude = np.iinfo(np.int16).max
clipped = (data * amplitude).astype(np.int16)

fig, ax = plt.subplots(figsize=(16, 10), layout='tight')

ax.scatter(
    fftfreq[:len(fft)//2],
    fft[:len(fft)//2],
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

plt.show()

write(
    r'audio/' + f"{new_filename[5:-3]}.wav",
    44100,
    clipped
)
