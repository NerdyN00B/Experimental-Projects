import numpy as np
import matplotlib.pyplot as plt
import time

from mydaq import MyDAQ

time.sleep(20)

now = time.strftime("%Y%m%d%H%M%S")

frequencies = [300, 500, 1000, 2000, 5000, 10000, 15000]
amplitude = 3  # Volume control
repeat = 1

daq = MyDAQ(44100, 'myDAQ4')

data = daq.measure_spectrum(
    frequencies=frequencies,
    amplitude=amplitude,
    duration=2,
    repeat=repeat,
)

np.save(f'data/transfer_measurement_{now}.npy', data)

full_transfer = daq.get_transfer_functions(data, frequencies, repeat=repeat)
db, std_db, phase, std_phase = daq.analyse_transfer(full_transfer)