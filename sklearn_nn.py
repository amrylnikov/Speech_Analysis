import librosa
import soundfile
import os, glob, pickle
import numpy as np

class Sklearn_NN():
    def extract_feature(self, file_name, mfcc, chroma, mel):
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate=sound_file.samplerate
            if chroma:
                stft=np.abs(librosa.stft(X))
            result=np.array([])
            if mfcc:
                mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result=np.hstack((result, mfccs))
            if chroma:
                chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
                result=np.hstack((result, chroma))
            if mel:
                mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
                result=np.hstack((result, mel))
        return result

    def load_data2(self, file2):
        x = []
        for file in glob.glob(file2):
            file_name=os.path.basename(file)
            feature=self.extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
        return np.array(x)

    def action(self, file):
        with open('Neural_Network\\model_pickle', 'rb') as f:
            mod = pickle.load(f)
        x_train2 = self.load_data2(file)
        print((x_train2))
        print((len(x_train2)))

        m = mod.predict(x_train2)
        print(m)
        return m

if __name__ == '__main__':
    filename = "C:/Users/Алексей/PycharmProjects/try3/venv/Include/Audio/5_sec.wav"

    Sklearn_NN = Sklearn_NN()
    Sklearn_NN.action(filename)