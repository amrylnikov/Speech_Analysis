import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import math


debussy_file = "test122.wav"
debussy, sr = librosa.load(debussy_file)
FRAME_SIZE = 2048
HOP_LENGTH  = 512

sc_debussy = librosa.feature.spectral_centroid(y=debussy, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
frames = range(len(sc_debussy))
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)


ban_debussy = librosa.feature.spectral_bandwidth(y=debussy, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

plt.figure(figsize=(25,10))

#ax1 = plt.subplot(2, 1, 1)
plt.plot(t, sc_debussy, color='b')

#ax2 = plt.subplot(2, 1, 2)
plt.plot(t, ban_debussy, color='g')
plt.show()