import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci
import math

# Import data

file = r'data\20251126113239_4cmdictaat.npy'
now = file.strip('.npy').split('_')[-1]

data = np.load(file)

data -= np.mean(data)

# Initiate useful variables and arrays

# sample_rate = 44100

# time_arr = np.linspace(0,4, sample_rate)
# freqs = [300, 500, 700, 50, 900, 10]
# amps = [20, 20, 20, 50, 20, 150]

# # print(time_arr)
# #print(np.array([freqs, amps]).T)

# # Create a standard sound signal with some peaks

# def signal(freqs, amps):
#     signal = np.zeros(len(time_arr))
#     for freq, amp in np.array([freqs, amps]).T:
#         sin = amp * np.sin(time_arr*freq)
#         signal += sin
#     return signal

# sign = signal(freqs, amps)

# Create a gaussian window to convolve with
Wind_rad = int(44100 / 1000 * 6)
Wind_range = np.linspace(-3, 3, Wind_rad)
Window = sci.norm.pdf(Wind_range)

# Convolve signal and gaussian

Window = np.array(Window)

Normalizer = np.convolve(abs(data), Window)

# Divide signal by convolved array

Normalizer[0] = 1

Norm_sign = data / Normalizer[math.floor(Wind_rad/2):len(data) + math.floor(Wind_rad/2)]

Norm_sign /= np.max(Norm_sign) / 10

# Plot new signal

plt.plot(Norm_sign * Wind_rad*5, label = 'Normalized signal')
plt.plot(Normalizer, label = 'Normalizer')
plt.plot(data, label = 'signal')
plt.legend()
plt.xlim(0, len(data) / 5)
plt.ylim(-1, 1)

plt.show()