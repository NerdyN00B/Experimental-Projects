from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import os

dir = r'D:\School\Natuurkunde WO\Experimental-Projects\audio\testables'

for file in os.listdir(dir):
    if file.endswith('.wav'):
        filepath = os.path.join(dir, file)
        
        sample_rate, data = read(filepath)
        
        # print(file)
        # print(sample_rate)

        np.save(file.replace('.wav', '.npy'), data)