import numpy as np
import matplotlib.pyplot as plt
import time

from mydaq import MyDAQ

now = time.strftime("%Y%m%d%H%M%S")

daq = MyDAQ(44100, 'myDAQ2')

_, sine = MyDAQ.generateWaveform('sine', 44100, 1984, duration=5, amplitude=3)

frequencies = [500, 600, 700, 800, 900, 1000] #, 5000, 10000, 15000]
for freq in frequencies:
    _, newsine = MyDAQ.generateWaveform('sine', 44100, freq, duration=5, amplitude=3)
    sine += newsine

sine /= np.max(np.abs(sine))  # Normalize to -1 to 1
# sine *= 2  # Scale sine sum

data = daq.readWrite(sine, write_channel='ao0', read_channel='ai0')
np.save(f"data/{now}_multiple_freq.npy", data)

fft = np.fft.fft(data)
fft_freq = np.fft.fftfreq(len(data), 1/daq.samplerate)
dB = 20*np.log10(np.abs(fft))

fig, ax = plt.subplots(figsize=(16, 10), layout='tight')

frequencies.append(1984)
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

ax.vlines(
    frequencies,
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
ax.set_xlabel('Frequency (Hz)', fontsize=24)
ax.set_ylabel('Magnitude (dB)', fontsize=24)
ax.set_title(f'Multiple Frequencies', fontsize=28)
ax.tick_params(labelsize = 16)
ax.grid()



plt.savefig(f'figures/{now}_multiple_frequencies_Hz.png', dpi=300)
plt.savefig(f'figures/{now}_multiple_frequencies_Hz.pdf', dpi=300)

