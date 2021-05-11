import pandas as pd
import numpy as np

import os
import sys
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import sounddevice
from scipy.fftpack import fft
from scipy.io.wavfile import write

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


class Keras_NN():
    def __init__(self):
        self.new_model = tf.keras.models.load_model('Neural_Network/keras.h5')

    def noise(self, data):
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data

    def stretch(self, data, rate=0.8):
        return librosa.effects.time_stretch(data, rate)

    def shift(self, data):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)

    def pitch(self, data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


    def extract_features(self, data, sample_rate):
        # ZCR
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result = np.hstack((result, zcr))  # stacking horizontally

        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft))  # stacking horizontally

        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc))  # stacking horizontally

        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms))  # stacking horizontally

        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))  # stacking horizontally

        return result


    def get_features(self, path):
        data, sample_rate = librosa.load(path)
        print(data.shape)
        # without augmentation
        res1 = self.extract_features(data, sample_rate)
        result = np.array(res1)

        # data with noise
        noise_data = self.noise(data)
        res2 = self.extract_features(noise_data, sample_rate)
        result = np.vstack((result, res2))  # stacking vertically

        # data with stretching and pitching
        new_data = self.stretch(data)
        data_stretch_pitch = self.pitch(new_data, sample_rate)
        res3 = self.extract_features(data_stretch_pitch, sample_rate)
        result = np.vstack((result, res3))  # stacking vertically

        return result

    def get_features_with_data(self, data):
        sample_rate = 44100

        # without augmentation
        res1 = self.extract_features(data, sample_rate)
        result = np.array(res1)

        # data with noise
        noise_data = self.noise(data)
        res2 = self.extract_features(noise_data, sample_rate)
        result = np.vstack((result, res2))  # stacking vertically

        # data with stretching and pitching
        new_data = self.stretch(data)
        data_stretch_pitch = self.pitch(new_data, sample_rate)
        res3 = self.extract_features(data_stretch_pitch, sample_rate)
        result = np.vstack((result, res3))  # stacking vertically

        return result

    def keras_action(self, filename):
        features = pd.read_csv("Neural_Network/features.csv")

        X = features.iloc[: ,:-1].values
        Y = features['labels'].values
        print(Y.shape)

        # As this is a multiclass classification problem onehotencoding our Y.
        encoder = OneHotEncoder()
        Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
        print("cool")

        #get date from file
        feature = self.get_features(filename)
        feature = np.expand_dims(feature, axis=2)

        #running model
        pred_test = self.new_model.predict(feature)
        y_pred = encoder.inverse_transform(pred_test)
        print(y_pred[0][0])
        return y_pred[0][0]

    def clean_dataset(self, df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    def keras_action_with_data(self, data):
        features = pd.read_csv("Neural_Network/features.csv")
        data5 = np.ndarray(shape= (len(data)))
        z = 0
        s = 0
        for i in range(len(data)):
            if pd.isna(data[i]):
                s = s + 1
            elif (data[i] == 0.):
                data5[z] = 0.000152587890625
                z = z + 1
            elif (data[i] > 3.4028235e+08):
                data5[z] = 3.4028235e+08
                z = z + 1
            elif (data[i] < -3.4028235e+08):
                data5[z] = -3.4028235e+08
                z = z + 1
            else:
                data5[z] = data[i]
                z = z + 1
        data6 = np.ndarray(shape=(len(data5)-s), dtype = float)

        for i in range(len(data6)):
            data6[i] = data5[i]

        Y = features['labels'].values

        encoder = OneHotEncoder()
        Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
        print("keras_action is running")

        #get date from file
        feature = self.get_features_with_data(data6)
        feature = np.expand_dims(feature, axis=2)

        pred_test = self.new_model.predict(feature)
        y_pred = encoder.inverse_transform(pred_test)
        print(y_pred[0][0])
        return y_pred[0][0]


if __name__ == '__main__':
    record = sounddevice.rec(int(2 * 44100), samplerate=44100, channels=1)
    print(record.shape)
    m = np.squeeze(record)
    print(m.shape)
    Keras_NN = Keras_NN()
    a = Keras_NN.keras_action_with_data(m)