import numpy as np
import nidaqmx as dx
from time import sleep
from scipy.signal import sawtooth, square
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class DualDaq():
    """A class to controll the MyDAQ"""
    def __init__(self, samplerate: int):
        self.finite = dx.constants.AcquisitionType.FINITE
        self.__samplerate = samplerate

    @property
    def samplerate(self) -> int:
        return self.__samplerate

    @samplerate.setter
    def samplerate(self, new_samplerate: int) -> None:
        assert isinstance(new_samplerate, int), "Samplerate should be an integer."
        assert new_samplerate > 0, "Samplerate should be positive."
        self.__samplerate = new_samplerate

    @staticmethod
    def convertDurationToSamples(samplerate: int, duration: float) -> int:
        samples = duration * samplerate

        # Round down to nearest integer
        return int(samples)

    @staticmethod
    def convertSamplesToDuration(samplerate: int, samples: int) -> float:
        duration = samples / samplerate

        return duration

    @staticmethod
    def getTimeArray(duration: float, samplerate: int) -> np.ndarray:
        steps = DualDaq.convertDurationToSamples(samplerate, duration)
        return np.linspace(1 / samplerate, duration, steps)
    
    def readwritedual(self, write_data, write_channel, read_channel):
        samples = len(write_data)

        with dx.Task() as write_task, dx.Task() as read_task:
            write_task.ao_channels.add_ao_voltage_chan(write_channel)
            read_task.ai_channels.add_ai_voltage_chan(read_channel)

            write_task.timing.cfg_samp_clk_timing(
                rate=self.samplerate,
                sample_mode=self.finite,
                samps_per_chan=samples
            )

            read_task.timing.cfg_samp_clk_timing(
                rate=self.samplerate,
                sample_mode=self.finite,
                samps_per_chan=samples
            )

            write_task.write(write_data, auto_start=True)

            data = read_task.read(number_of_samples_per_channel=samples)
        return np.array(data)
    
    def gensine(self, frequency, duration, amplitude=1, offset=0):
        t = self.getTimeArray(duration, self.samplerate)
        return amplitude * np.sin(2 * np.pi * frequency * t) + offset