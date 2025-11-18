import nidaqmx as dx
import numpy as np

class MyDAQ_Long():
	def __init__(self, chunksize=1000):
		self.chunksize = chunksize
		self.data = []

	def capture(self, samplerate=44100, channel='myDAQ4/ai0'):
		"""Captures input of Ai0 for duration seconds at samplerate samplerate."""
		self.data = [] # Clear data
		
		self.task = dx.Task()
		self.task.ai_channels.add_ai_voltage_chan(channel)

		self.task.timing.cfg_samp_clk_timing(
            samplerate,
            sample_mode=dx.constants.AcquisitionType.CONTINUOUS
        )
		self.task.register_every_n_samples_acquired_into_buffer_event(
            self.chunksize,
            self._updateData
        )

		self.task.start()
		print("Recording...")
		input("Press enter to stop.")
		self.task.stop()
	
		return np.array(self.data).flatten()

	def _updateData(self,
                    task_handle,
                    every_n_samples_event_type,
                    number_of_samples,
                    callback_data
                    ):
		newdata = self.task.read(number_of_samples_per_channel=self.chunksize)
		self.data.append(newdata)
		return 0