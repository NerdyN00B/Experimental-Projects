from scipy.io.wavfile import write
import numpy as np


samplerate = 44100  # Hertz

file = r'data/hallo_ep_2.npy'

data = np.load(file)

data -= np.mean(data)  # Remove DC offset
data = data / np.max(np.abs(data))  # Normalize to -1 to 1

amplitude = np.iinfo(np.int16).max
data = (data * amplitude).astype(np.int16)

write(file.replace('.npy', '.wav').replace('data', 'audio'), samplerate, data)