import numpy as np

from mydaq import MyDAQ

daq = MyDAQ(44100, 'myDAQ4')

file = r'data/screaming_into_glass.npy'

data = daq.read(10)

np.save(file, data)