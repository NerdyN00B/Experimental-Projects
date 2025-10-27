import pystoi as stoi
import soundfile as sf
import numpy as np

def stoi2sti(stoi):
    exp = np.exp(-13.19 * stoi + 6.52)
    return 100 / (1 + exp)

clean, fs = sf.read(r"audio/hallo_ep_2.wav")
noisy, fs = sf.read(r"audio\20251027161549_playback_record.wav")

d = stoi.stoi(clean, noisy, fs, extended=False)

print(d)
print(stoi2sti(d))

# First measurement gets STOI score of 0.4197151258966351
# Last measurement gets STOI score of 0.399973658959185