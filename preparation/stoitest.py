import pystoi as stoi
import soundfile as sf

clean, fs = sf.read(r"audio/hallo_ep_2.wav")
noisy, fs = sf.read(r"audio\20251022164245_playback_record.wav")

d = stoi.stoi(clean, noisy, fs, extended=False)

print(d)

# First measurement gets STOI score of 0.4197151258966351
# Last measurement gets STOI score of 0.399973658959185