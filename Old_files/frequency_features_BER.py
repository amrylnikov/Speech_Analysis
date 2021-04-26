import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import math


debussy_file = "test122.wav"
debussy, sr = librosa.load(debussy_file)
FRAME_SIZE = 2048
HOP_SIZE = 512

debussy_spec = librosa.stft(debussy, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

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
        #print(sum_power_low_frequencies)
        #print(sum_power_high_frequencies)
        ber_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(ber_current_frame)

    return np.array(band_energy_ratio)

ber_debussy = calculate_band_energy_ratio(debussy_spec, 2000, sr)

frames = range(len(ber_debussy))
t = librosa.frames_to_time(frames, hop_length=HOP_SIZE)


plt.figure(figsize=(25, 10))

plt.plot(t, ber_debussy, color="b")
plt.ylim((0, 20000))
plt.show()