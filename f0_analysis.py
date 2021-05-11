import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import scipy.io.wavfile as wavfile
import scipy.fftpack as fftpk
import librosa
import librosa.display
import IPython.display as ipd
from aubio import source, pitch

class F0_Analyser():
    def f0_analysis(self, audio_path, praat_path):
        audio_path = "Audio/" + os.path.basename(audio_path)
        from aubio import source, pitch
        print(audio_path)
        # load audio
        signal = basic.SignalObj(audio_path)
        filename = audio_path
        debussy, sr = librosa.load(filename)
        FRAME_SIZE = 1024
        HOP_LENGTH = 512

        # Frequency
        FFT = abs(fftpk.fft(debussy))
        fregs = fftpk.fftfreq(len(FFT), (1.0/sr))

        # Spectrogram
        S_debussy = librosa.stft(debussy, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
        Y_debussy = librosa.power_to_db(np.abs(S_debussy) ** 2)

        # Time domain features AE, RMS, ZCR part
        sample_duration = 1 / sr
        tot_samples = len(debussy)
        duration = 1 / sr * tot_samples
        zcr_debussy = librosa.feature.zero_crossing_rate(debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        rms_debussy = librosa.feature.rms(debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]


        def amplitude_envelope(signal, frame_size, hop_length):
            """Calculate the amplitude envelope of a signal with a given frame size nad hop length."""
            amplitude_envelope = []

            # calculate amplitude envelope for each frame
            for i in range(0, len(signal), hop_length):
                amplitude_envelope_current_frame = max(signal[i:i + frame_size])
                amplitude_envelope.append(amplitude_envelope_current_frame)

            return np.array(amplitude_envelope)

        ae_debussy = amplitude_envelope(debussy, FRAME_SIZE, HOP_LENGTH)
        frames = range(len(ae_debussy))
        t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
        frames2 = range(len(rms_debussy))
        t2 = librosa.frames_to_time(frames2, hop_length=HOP_LENGTH)

        # Frequency features BER part

        debussy_spec = librosa.stft(debussy, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)

        def calculate_split_frequency_bin(spectrogram, split_frequency, sample_rate):
            """Infer the frequency bin associated to a given split frequency."""

            frequency_range = sample_rate / 2
            frequency_delta_per_bin = frequency_range / spectrogram.shape[0]
            split_frequency_bin = np.floor(split_frequency / frequency_delta_per_bin)
            return int(split_frequency_bin)

        split_frequency_bin = calculate_split_frequency_bin(debussy_spec, 2000, 22050)

        def calculate_band_energy_ratio(spectrogram, split_frequency, sample_rate):
            """Calculate band energy ratio with a given split frequency."""

            split_frequency_bin = calculate_split_frequency_bin(spectrogram, split_frequency, sample_rate)
            band_energy_ratio = []

            # calculate power spectrogram
            power_spec = np.abs(spectrogram) ** 2
            power_spec = power_spec.T

            # calculate BER value for each frame
            for frequencies_in_frame in power_spec:
                sum_power_low_frequencies = np.sum(frequencies_in_frame[:split_frequency_bin])
                sum_power_high_frequencies = np.sum(frequencies_in_frame[split_frequency_bin:])
                if sum_power_high_frequencies == 0.0:
                    sum_power_high_frequencies = 0.00000000000000001
                ber_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
                band_energy_ratio.append(ber_current_frame)

            return np.array(band_energy_ratio)

        ber_debussy = calculate_band_energy_ratio(debussy_spec, 2000, sr)

        frames3 = range(len(ber_debussy))
        t3 = librosa.frames_to_time(frames3, hop_length=HOP_LENGTH)

        sc_debussy = librosa.feature.spectral_centroid(y=debussy, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        ban_debussy = librosa.feature.spectral_bandwidth(y=debussy, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        frames4 = range(len(sc_debussy))
        t4 = librosa.frames_to_time(frames4, hop_length=HOP_LENGTH)

        # YAAPT pitches
        pitchY = pYAAPT.yaapt(signal, frame_length=40, tda_frame_length=40, f0_min=75, f0_max=600)

        # YIN pitches
        downsample = 1
        samplerate = 0
        win_s = 1764 // downsample  # fft size
        hop_s = 441 // downsample  # hop size
        s = source(filename, samplerate, hop_s)
        samplerate = s.samplerate
        tolerance = 0.8
        pitch_o = pitch("yin", win_s, hop_s, samplerate)
        pitch_o.set_unit("midi")
        pitch_o.set_tolerance(tolerance)

        pitchesYIN = []
        confidences = []

        total_frames = 0

        while True:
            samples, read = s()
            pitch = pitch_o(samples)[0]
            pitch = int(round(pitch))
            confidence = pitch_o.get_confidence()
            pitchesYIN += [pitch]
            confidences += [confidence]
            total_frames += read
            if read < hop_s:
                break

        # PRAAT
        praat = np.genfromtxt(praat_path, filling_values=0)
        praat = praat[:, 1]

        # plot
        plt.figure(figsize=(19, 15))

        ax1 = plt.subplot(3, 4, 6)
        plt.plot(np.asarray(pitchesYIN), color='green')
        plt.title("YIN")
        plt.ylim(0, 500)
        ax2 = plt.subplot(3, 4, 10)
        plt.plot(pitchY.samp_values, color='green')
        plt.title("YAAPT")
        plt.ylim(0, 500)
        ax3 = plt.subplot(3, 4, 3)
        plt.plot(t, ae_debussy, color="black")
        #plt.ylim(0, 1)
        plt.title("Amplitude Envelope (AE)")
        ax4 = plt.subplot(3, 4, 7)
        plt.plot(t2, rms_debussy, color="black")
        #plt.ylim(0, 1)
        plt.title("Root-Mean-Square energy (RMS)")
        ax5 = plt.subplot(3, 4, 11)
        plt.plot(t2, zcr_debussy, color="black")
        #plt.ylim(0, 1)
        plt.title("Zero-Crossing Rate (ZCR)")
        ax6 = plt.subplot(3, 4, 4)
        plt.plot(t3, ber_debussy, color="b")
        plt.title("Band Energy Ratio (BER)")
        ax7 = plt.subplot(3, 4, 8)
        plt.plot(t4, sc_debussy, color="b")
        plt.title("Spectral Centroid (SC)")
        ax8 = plt.subplot(3, 4, 12)
        plt.plot(t4, ban_debussy, color="b")
        plt.title("Bandwidth (BW)")
        ax9 = plt.subplot(3, 4, 2)
        plt.plot(praat, color='red')
        plt.title("Praat")
        plt.ylim(0, 500)
        ax10 = plt.subplot(3, 4, 1)
        librosa.display.waveplot(debussy, x_axis= "off")
        plt.title("Audio wave")
        ax11 = plt.subplot(3, 4, 5)
        plt.plot(fregs[range(len(FFT) // 2)], FFT[range(len(FFT) // 2)])
        plt.title("Frequency")
        ax12 = plt.subplot(3, 4, 9)
        librosa.display.specshow(Y_debussy, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.f")
        plt.title("Spectrogram")
        plt.show()

if __name__ == '__main__':
    audio = "Audio/Actor_01/03-01-01-01-01-01-01.wav"
    praat = "Praat/Actor_01/03-01-01-01-01-01-01.txt"
    F0 = F0_Analizer()
    F0.f0_analysis(audio, praat)