import numpy as np
import matplotlib.pyplot as plt

from mydaq import MyDAQ

daq = MyDAQ(44100, 'myDAQ4')

file = r'data/hallo_ep_2.npy'

print("Recording...")

data = daq.read(4, channel='ai1')

np.save(file, data)

fig, ax = plt.subplots(2, 1, figsize=(16, 10), layout='tight')

ax[0].plot(data)
ax[1].plot(np.abs(np.fft.fft(data))[:len(data)//2])
ax[1].set(xscale='log', yscale='log')

plt.show()