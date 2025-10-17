import numpy as np
import matplotlib.pyplot as plt
import time

from mydaq import MyDAQ

just_playback = False

playback = np.load(r'data\hallo_ep_2.npy')

now = time.strftime("%Y%m%d%H%M%S")

mean = np.mean(playback)
playback -= mean
playback /= np.max(np.abs(playback))

daq = MyDAQ(44100, 'myDAQ4')

if just_playback:
    daq.write(playback*3) # Volume control
else:
    data = daq.readWrite(playback*3, write_channel='ao0', read_channel='ai1')
    np.save(f'data/{now}_playback_record.npy', data)

    fft = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), 1/daq.samplerate)
    dB = 20*np.log10(np.abs(fft))

    fig, ax = plt.subplots(figsize=(16, 10), layout='tight')

    ax.scatter(
        fft_freq[:len(fft)//2],
        dB[:len(fft)//2],
        s=1,
        marker='.',
        color='k',
    )

    ax.set(
        xscale='log',
        xlabel='Frequency (Hz)',
        ylabel='Magnitude (dB)',
        title=f'playback and record',
    )

    plt.savefig(f'figures/{now}_playback_recording.png', dpi=300)
    plt.savefig(f'figures/{now}_playback_recording.pdf', dpi=300)
