import wave
import numpy as np
import sys
import matplotlib.pyplot as plt


class Analysis():
    def simple_plot(self, pathb):
        print(pathb)
        wav = wave.open(pathb, "r")
        raw = wav.readframes(-1)
        raw = np.frombuffer(raw, "int16")
        sampleRate = wav.getframerate()

        if wav.getnchannels() == 2:
            print("Dude, uncool")
            sys.exit(0)

        Time = np.linspace(0, len(raw) / sampleRate, num=len(raw))
        plt.title("Waveform")
        plt.plot(Time, raw, color="blue")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

if __name__ == '__main__':
    Analysis()