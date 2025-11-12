import numpy as np
import matplotlib.pyplot as plt
import time

from dualdaq import DualDaq

amplitude = 0.5
duration = 1.1

freqs = np.logspace(np.log10(400), np.log10(14000), 20, dtype=int)
print(freqs)

daq = DualDaq(44100)


transfers = []
for i, freq in enumerate(freqs):
    print(f'{freq}, {i+1} / {len(freqs)}')
    sine = daq.gensine(freq, duration, amplitude)
    data = daq.readwritedual(
        sine,
        'myDAQ2/ao0',
        'myDAQ1/ai0',
    )

    fft = np.fft.fft(data[:int(-0.1*daq.samplerate)])
    transfers.append(fft)

now = time.strftime('%Y%m%d%H%M%S')
np.save(f'data/{now}_transfer.npy', np.asarray(transfers))
fftfreq = np.fft.fftfreq(int((duration - 0.1) * daq.samplerate), 1/daq.samplerate)

fig, ax = plt.subplots(4, 5, figsize=(16,10), layout='tight')

for i, transfer in enumerate(transfers):
    x = i%5
    y = i//5

    db = 20*np.log10(np.abs(transfer))
    ax[y, x].scatter(
        fftfreq[:len(fftfreq)//2],
        db[:len(db)//2],
        s=10,
        marker='.',
        c='k',
    )
    
    idx = np.argmin(np.abs(fftfreq - freqs[i]))

    ax[y, x].vlines(
        fftfreq[idx],
        np.min(db),
        np.max(db),
        colors='r',
        linestyles='dashed',
        alpha=0.5,
        linewidth=1,
    )
    
    ax[y, x].set(
        xlabel='Frequency (Hz)',
        ylabel='Magnitude (dB)',
        title=f'Transfer {freqs[i]} Hz',
    )
    
    ax[y, x].set_xlim(freqs[i]-50, freqs[i]+50)
    print(db[idx-5:idx+5])

fig.savefig(f'figures/{now}_transfer_overview.png', dpi=300)
