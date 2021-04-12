import wave
import numpy as np
import sys
import matplotlib.pyplot as plt


class Analysis():
    #def __init__(self):
        #self.pathA = "delme_rec_unlimited_rd_y_na1.wav"

        #self.pathA = "delme_rec_unlimited_rd_y_na1.wav"
        #self.simple_plot(self.pathA)

    def simple_plot(self, pathb):
        print(pathb)
        self.wav = wave.open(pathb, "r")
        self.raw = self.wav.readframes(-1)
        self.raw = np.frombuffer(self.raw, "int16")
        self.sampleRate = self.wav.getframerate()

        if self.wav.getnchannels() == 2:
            print("Dude, uncool")
            sys.exit(0)

        self.Time = np.linspace(0, len(self.raw) / self.sampleRate, num=len(self.raw))
        plt.title("Waveform")
        plt.plot(self.Time, self.raw, color="blue")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

if __name__ == '__main__':
    Analysis()