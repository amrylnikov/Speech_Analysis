import matplotlib.pyplot as plt
import scipy
import wave
import scipy.io.wavfile as wavfile
import scipy.fftpack as fftpk
import numpy as np

class FFT():
    def FFT_plot(self, path):
        s_rate, signa = wavfile.read(path) #сам сигнал и частота дискретизации
        wav = wave.open(path, "r")
        raw = wav.readframes(-1)
        raw = np.frombuffer(raw, "int16")
        sampleRate = wav.getframerate()

        Time = np.linspace(0, len(raw) / sampleRate, num=len(raw))
        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(Time, raw)
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Loudness')

        FFT = abs(fftpk.fft(signa))
        fregs = fftpk.fftfreq(len(FFT), (1.0/s_rate))

        #frequency
        ax2.plot(fregs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])
        ax2.set_xlabel('Frequency(Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Spectrum')

        plt.show()

if __name__ == '__main__':
    FFT()