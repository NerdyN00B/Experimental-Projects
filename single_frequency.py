import numpy as np
import matplotlib.pyplot as plt
import time

from mydaq import MyDAQ


frequency = 500  # Hz

now = time.strftime("%Y%m%d%H%M%S")


daq = MyDAQ(200000, 'myDAQ1')

sine = MyDAQ.generateWaveform(
    'sine',
    samplerate=daq.samplerate,
    frequency=frequency,
    amplitude=4,
    duration=5,
    )

data = daq.readWrite(sine)
np.save(f'data/single_frequency_{frequency}Hz_{now}.npy', data)

fft = np.fft.fft(data)
fft_freq = np.fft.fftfreq(len(data), 1/daq.samplerate)

fig, ax = plt.subplots()
ax.scatter(
    fft_freq[:len(fft)//2],
    20*np.log10(np.abs(fft)[:len(fft)//2]),
    s=1,
    marker='.',
    )

idx = MyDAQ.find_nearest_idx(fft_freq, frequency)
ax.scatter(
    fft_freq[idx],
    20*np.log10(np.abs(fft)[idx]),
    color='red',
    marker='star',    
)
