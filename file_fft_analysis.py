import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.fftpack import fft
from math import pi

plt.close('all')

Fs = 16000
d = 3

print("говори")

a = sd.rec(int(d*Fs), Fs, 1, blocking = 'True')

print('закончил')

sd.play(a, Fs)
plt.plot(a)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Loudness')

x_f = fft(a)

n = np.size(a)
fr = (Fs/2)*np.linspace(0,1,round(n/2))
x_m = (2/n)*abs(x_f[0:np.size(fr)])

#spectrum
plt.figure()
plt.plot(fr, x_m)
plt.xlabel('Frequency(Hz)')
plt.ylabel('Magnitude')
plt.title('Spectrum')

plt.show()