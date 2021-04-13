import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

y, sr = librosa.load(librosa.ex('trumpet'))
D = librosa.stft(y)  # STFT of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

C = librosa.cqt(y=y, sr=sr)
C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

Sa = librosa.note_to_hz('F4')
fig, (ax1, ax2) = plt.subplots(2, 1)
librosa.display.specshow(C_db, y_axis='cqt_svara', Sa=Sa, x_axis='time', ax=ax1)
ax1.set(title='Hindustani decoration',
       ylim=[Sa, 2*Sa])

librosa.display.specshow(C_db, y_axis='cqt_svara', Sa=Sa, mela=22, x_axis='time', ax=ax2)
ax2.set(title='Carnatic decoration',
       ylim=[Sa, 2*Sa])
plt.show()