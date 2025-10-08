import numpy as np
import matplotlib.pyplot as plt
import time

from mydaq import MyDAQ

now = time.strftime("%Y%m%d%H%M%S")

daq = MyDAQ(44100, 'myDAQ4')

_, sine = MyDAQ.generateWaveform('sine', 44100, 300, duration=5, amplitude=3)

frequencies = [500, 1000, 2000, 5000, 10000, 15000]
for freq in frequencies:
    _, newsine = MyDAQ.generateWaveform('sine', 44100, freq, duration=5, amplitude=3)
    sine += newsine

sine /= np.max(np.abs(sine))  # Normalize to -1 to 1
sine *= 3  # Scale to -3 to 3 V

data = daq.readWrite(sine)
np.save(f"data/multiple_freq_{now}.npy", data)

fft = np.fft.fft(data)
fft_freq = np.fft.fftfreq(len(data), 1/daq.samplerate)
dB = 20*np.log10(np.abs(fft))

fig, ax = plt.subplots(figsize=(16, 10))

frequencies.append(300)
# idx = MyDAQ.find_nearest_idx(fft_freq, frequencies)
# ax.vlines(
#     fft_freq[idx],
#     ymin=np.min(dB),
#     ymax=np.max(dB),
#     colors='r',
#     linestyles='dashed',
#     alpha=0.5,
#     linewidth=1,
# )

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
    title=f'Multiple Frequencies'
)

plt.savefig(f'figures/multiple_frequencies_Hz_{now}.png', dpi=300)
plt.savefig(f'figures/multiple_frequencies_Hz_{now}.pdf', dpi=300)

