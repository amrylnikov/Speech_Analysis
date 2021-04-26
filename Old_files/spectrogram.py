import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt



FRAME_SIZE = 2048
HOP_SIZE = 512
filename = "debussy.wav"
debussy, sr = librosa.load(filename)

S_debussy = librosa.stft(debussy, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
Y_debussy = librosa.power_to_db(np.abs(S_debussy) ** 2)
plt.figure(figsize=(25, 10))
librosa.display.specshow(Y_debussy, sr=sr, hop_length=HOP_SIZE, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.f")
plt.show()