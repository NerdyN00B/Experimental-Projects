import numpy as np
import matplotlib.pyplot as plt
import time

from dualdaq import DualDaq

amplitude = 0.5
duration = 1.1

freqs = np.logspace(np.log10(400), np.log10(14000), 20, dtype=int)
print(freqs)

daq = DualDaq(44100)


transfers = []
for i, freq in enumerate(freqs):
    print(f'{freq}, {i+1} / {len(freqs)}')
    sine = daq.gensine(freq, duration, amplitude)
    data = daq.readwritedual(
        sine,
        'myDAQ2/ao0',
        'myDAQ1/ai0',
    )

    fft = np.fft.fft(data[:int(-0.1*daq.samplerate)])
    transfers.append(fft)

fig, ax = plt.subplots(4, 5, figsize=(16,10), layout='tight')
