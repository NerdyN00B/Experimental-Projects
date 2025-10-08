import numpy as np

from mydaq import MyDAQ

data = np.load(r'data\screaming_into_glass.npy')

mean = np.mean(data)
data -= mean
data /= np.max(np.abs(data))

daq = MyDAQ(44100, 'myDAQ4')
daq.write(data*0.25) # Volume control