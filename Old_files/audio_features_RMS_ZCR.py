import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd

# load audio
debussy_file = "for_f0.wav"

debussy, sr = librosa.load(debussy_file)

FRAME_SIZE = 1024
HOP_LENGTH = 512


rms_debussy = librosa.feature.rms(debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

frames = range(len(rms_debussy))
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

# ZCR

zcr_debussy = librosa.feature.zero_crossing_rate(debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

# rms energy is graphed in red

plt.figure(figsize=(15, 17))

ax = plt.subplot(2, 1, 1)
librosa.display.waveplot(debussy, alpha=0.5)
plt.plot(t, rms_debussy, color="r")
plt.ylim((-1, 1))
plt.title("RMS")

ax = plt.subplot(2, 1, 2)
plt.plot(t, zcr_debussy, color="y")
plt.ylim(0, 1)
plt.title("ZCR")

plt.show()

