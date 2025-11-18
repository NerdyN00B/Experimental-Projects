import numpy as np
import time

from scipy.io.wavfile import write
from longdaq import MyDAQ_Long

now  = time.strftime("%Y%m%d%H%M%S")

filename = 'dictee_sam'

daq = MyDAQ_Long()

data = daq.capture(channel='myDAQ1/ai0')

np.save(r'data/' + f"{now}_{filename}.npy", data)

data -= np.mean(data)  # Remove DC offset
data = data / np.max(np.abs(data))  # Normalize to -1 to 1

amplitude = np.iinfo(np.int16).max
data = (data * amplitude).astype(np.int16)

write(
    r'audio/' + f"{now}_{filename}.wav",
    44100,
    data
)
