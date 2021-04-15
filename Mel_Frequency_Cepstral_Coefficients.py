import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

class MFCC():
    def MFCC_plot(self, path):
    # load audio files with librosa
        signal, sr = librosa.load(path) #сам сигнал и частота дискретизации
        print(signal.shape) # форма сигнала (?)

        # Extracting MFCCs
        mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
        print(mfccs.shape)
        """
        plt.figure(figsize=(25, 10))
        librosa.display.waveplot(signal,
                                 sr=sr,
                                 max_points=50000.0,
                                 x_axis='time',
                                 offset=0.0,
                                 max_sr=1000,
                                 ax=None,
                                 )
        plt.show()
        """
        # Visualising MFCCs
        #fig = plt.figure(figsize=(25, 10))
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        #librosa.display.specshow(mfccs, x_axis="time", ax=ax1, sr=sr)
        ax1.set(title='Коэффициенты MFCC по времени')
        img1 = librosa.display.specshow(mfccs, y_axis='frames', ax=ax1, sr=sr)
        fig.colorbar(img1, ax=ax1, format="%+2.f")

        # Computing first / second MFCCs derivatives
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        print(delta_mfccs.shape)

        ax2.set(title='Дельта MFCC')
        img2 = librosa.display.specshow(delta_mfccs, y_axis='frames', ax=ax2, sr=sr)
        fig.colorbar(img2, ax=ax2, format="%+2.f")

        ax3.set(title='Дельта дельты MFCC')
        img3 = librosa.display.specshow(delta2_mfccs,x_axis="time", y_axis='frames', ax=ax3, sr=sr)
        fig.colorbar(img3, ax=ax3, format="%+2.f")
        plt.show()

        mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
        print(mfccs_features.shape)

if __name__ == '__main__':
    MFCC()