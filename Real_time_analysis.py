"""
    Notebook for streaming data from a microphone in realtime
    audio is captured using pyaudio
    then converted from binary data to ints using struct
    then displayed using matplotlib
    scipy.fftpack computes the FFT
    if you don't have pyaudio, then run
    >>> pip install pyaudio
    note: with 2048 samples per chunk, I'm getting 20FPS
    when also running the spectrum, its about 15FPS
"""

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import pandas as pd
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import struct
import sounddevice
from scipy.fftpack import fft
from scipy.io.wavfile import write
from keras_nn import Keras_NN
import sys
import time

class AudioStream(object):
    def __init__(self):

        # stream constants
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.pause = False
        self.Keras_NN = Keras_NN()

        # stream object
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )
        self.init_plots()
        self.start_plot()

    def init_plots(self):

        # x variables for plotting
        x = np.arange(0, 2 * self.CHUNK, 2)
        xf = np.linspace(0, self.RATE, self.CHUNK)

        # create matplotlib figure and axes
        self.fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 7))
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)

        # create a line object with random data
        self.line, = ax1.plot(x, np.random.rand(self.CHUNK), '-', lw=2)

        # create semilogx line for spectrum
        self.line_fft, = ax2.semilogx(
            xf, np.random.rand(self.CHUNK), '-', lw=2)

        # format waveform axes
        ax1.set_title('AUDIO WAVEFORM')
        ax1.set_xlabel('samples')
        ax1.set_ylabel('volume')
        ax1.set_ylim(0, 255)
        ax1.set_xlim(0, 2 * self.CHUNK)
        plt.setp(
            ax1, yticks=[0, 128, 255],
            xticks=[0, self.CHUNK, 2 * self.CHUNK],
        )
        plt.setp(ax2, yticks=[0, 1],)

        # format spectrum axes
        ax2.set_title('AUDIO WAVEFORM')
        ax2.set_xlabel('frequency')
        ax2.set_ylabel('amplitude')
        ax2.set_xlim(20, self.RATE / 2)

        self.text = 'ЭМОЦИЯ'
        self.text2 = ax3.text(0.5, 0.5, self.text, horizontalalignment='center',
                 verticalalignment='center', transform=ax3.transAxes, fontsize=50, color='r')

        # show axes
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(5, 120, 1910, 1070)
        plt.show(block=False)

    def start_plot(self):

        print('stream started')
        frame_count = 1
        start_time = time.time()
        #record = sounddevice.rec(int(5*44100), samplerate=44100, channels=1)
        #sounddevice.wait()
        #write("output.wav", 44100, record)

        while not self.pause:
            if frame_count % 50 == 0:
                record = sounddevice.rec(int(3 * 44100), samplerate=44100, channels=1)
                print('+++')
                print(record)
                print('+++')
                m = np.squeeze(record)
                print('---')
                print(m)
                print('---')
                self.text = self.Keras_NN.keras_action_with_data(m)
            data = self.stream.read(self.CHUNK)
            #print(data)
            #print("---")
            data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)
            data_np = np.array(data_int, dtype='b')[::2] + 128
            self.line.set_ydata(data_np)
            #self.text = frame_count
            self.text2.set_text(self.text)
            # compute FFT and update line
            yf = fft(data_int)
            self.line_fft.set_ydata(
                np.abs(yf[0:self.CHUNK]) / (128 * self.CHUNK))
            # update figure canvas
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            frame_count += 1

        else:
            self.fr = frame_count / (time.time() - start_time)
            print('average frame rate = {:.0f} FPS'.format(self.fr))
            self.exit_app()

    def exit_app(self):
        print('stream closed')
        self.p.close(self.stream)

    def onClick(self, event):
        self.pause = True

if __name__ == '__main__':
    AudioStream()