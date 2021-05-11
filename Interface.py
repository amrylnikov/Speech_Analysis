from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QPushButton, QMessageBox
import sys
from Real_time_analysis import AudioStream
from pydub import AudioSegment
from file_analysis import Analysis
from linear_predictive_coding import LPC
from Mel_Frequency_Cepstral_Coefficients import MFCC
from file_fft_analysis import FFT
from sklearn_nn import Sklearn_NN
from keras_nn import Keras_NN
from f0_analysis import F0_Analyser
import sounddevice as sd
import soundfile as sf
import argparse
import tempfile
import queue

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.a = "3"
        self.path = ''
        self.praat_path = 'Praat/Praat_for_f0.txt'
        self.flag = True
        self.Analysis = Analysis()
        self.LPC = LPC()
        self.MFCC = MFCC()
        self.FFT = FFT()
        self.Sklearn_NN = Sklearn_NN()
        self.Keras_NN = Keras_NN()
        self.F0_Analyser = F0_Analyser()

        # Speech Load
        self.parser = argparse.ArgumentParser(add_help=False)
        self.parser.add_argument(
            '-l', '--list-devices', action='store_true',
            help='show list of audio devices and exit')
        self.args, self.remaining = self.parser.parse_known_args()
        if self.args.list_devices:
            print(sd.query_devices())
            self.parser.exit(0)
        self.parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            parents=[self.parser])
        self.parser.add_argument(
            'filename', nargs='?', metavar='FILENAME',
            help='audio file to store recording to')
        self.parser.add_argument(
            '-d', '--device', type=self.int_or_str,
            help='input device (numeric ID or substring)')
        self.parser.add_argument(
            '-r', '--samplerate', type=int, help='sampling rate')
        self.parser.add_argument(
            '-c', '--channels', type=int, default=1, help='number of input channels')
        self.parser.add_argument(
            '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
        self.args = self.parser.parse_args(self.remaining)
        self.q = queue.Queue()

        #Главное окно
        self.setWindowTitle("Анализ речи")
        self.setGeometry(50, 50, 530, 500)

        self.new_text = QtWidgets.QLabel(self)

        self.label = QtWidgets.QLabel(self)
        self.label.move(280, 55)
        self.label.setText("Запись голоса будет идти 5 секунд")
        self.label.adjustSize()

        self.label2 = QtWidgets.QLabel(self)
        self.label2.move(280, 105)
        self.label2.setText("Формат файла: .wav")
        self.label2.adjustSize()

        self.label3 = QtWidgets.QLabel(self)
        self.label3.move(280, 155)
        self.label3.setText("Визуализация громкости и частоты")
        self.label3.adjustSize()

        self.label4 = QtWidgets.QLabel(self)
        self.label4.move(280, 205)
        self.label4.setText("Анализ библиотекой wave")
        self.label4.adjustSize()

        self.label5 = QtWidgets.QLabel(self)
        self.label5.move(280, 255)
        self.label5.setText("Fast Fourier transform")
        self.label5.adjustSize()

        self.label6 = QtWidgets.QLabel(self)
        self.label6.move(280, 305)
        self.label6.setText("*Здесь будет указан ответ*")
        self.label6.adjustSize()

        self.label7 = QtWidgets.QLabel(self)
        self.label7.move(280, 355)
        self.label7.setText("*Здесь будет указан ответ*")
        self.label7.adjustSize()

        self.label8 = QtWidgets.QLabel(self)
        self.label8.move(280, 405)
        self.label8.setText("Используются 8 методов анализа")
        self.label8.adjustSize()

        self.btn = QtWidgets.QPushButton(self)
        self.btn.move(50,50)
        self.btn.setText("Запись речи")
        self.btn.setFixedWidth(200)
        self.btn.clicked.connect(self.speech_recording)

        self.btn2 = QtWidgets.QPushButton(self)
        self.btn2.move(50, 100)
        self.btn2.setText("Открыть файл")
        self.btn2.setFixedWidth(200)
        self.btn2.clicked.connect(self.get_path)

        self.btn3 = QtWidgets.QPushButton(self)
        self.btn3.move(50,150)
        self.btn3.setText("Анализ в реальном времени")
        self.btn3.setFixedWidth(200)
        self.btn3.clicked.connect(AudioStream)

        self.btn4 = QtWidgets.QPushButton(self)
        self.btn4.move(50, 200)
        self.btn4.setText("Запустить обычный анализ")
        self.btn4.setFixedWidth(200)
        self.btn4.clicked.connect(self.execute_file_analysis)

        self.btn5 = QtWidgets.QPushButton(self)
        self.btn5.move(50, 250)
        self.btn5.setText("Запустить анализ FFT")
        self.btn5.setFixedWidth(200)
        self.btn5.clicked.connect(self.execute_FFT)

        self.btn6 = QtWidgets.QPushButton(self)
        self.btn6.move(50, 300)
        self.btn6.setText("Эмоция по нейросети sklearn")
        self.btn6.setFixedWidth(200)
        self.btn6.clicked.connect(self.execute_Sklearn_NN)

        self.btn7 = QtWidgets.QPushButton(self)
        self.btn7.move(50, 350)
        self.btn7.setText("Эмоция по нейросети keras")
        self.btn7.setFixedWidth(200)
        self.btn7.clicked.connect(self.execute_Keras_NN)

        self.btn8 = QtWidgets.QPushButton(self)
        self.btn8.move(50, 400)
        self.btn8.setText("Анализ F0")
        self.btn8.setFixedWidth(200)
        self.btn8.clicked.connect(self.execute_F0)

    # просьба открыть файл
    def get_path(self):
        self.flag = True
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.path = QFileDialog.getOpenFileName(self,"Select Audio File",
                                                "",
                                                "All Files (*);;Audio File (*.wav)",
                                                options=options)
        self.path2 = self.path[0]

    def execute_file_analysis(self):
        if self.path == '':
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка!")
            msg.setText("Файл для анализа не выбран")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()
        else:
            print(self.path)
            if self.flag:
                self.Analysis.simple_plot(self.path[0])
            else:
                self.Analysis.simple_plot(self.path)

    def execute_LPC(self):
        if self.path == '':
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка!")
            msg.setText("Файл для анализа не выбран")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()
        else:
            print(self.path)
            if self.flag:
                self.LPC.LPC_plot(self.path[0])
            else:
                self.LPC.LPC_plot(self.path)

    def execute_MFCC(self):
        if self.path == '':
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка!")
            msg.setText("Файл для анализа не выбран")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()
        else:
            print(self.path)
            if self.flag:
                self.MFCC.MFCC_plot(self.path[0])
            else:
                self.MFCC.MFCC_plot(self.path)

    def execute_FFT(self):
        if self.path == '':
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка!")
            msg.setText("Файл для анализа не выбран")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()
        else:
            print(self.path)
            if self.flag:
                self.FFT.FFT_plot(self.path[0])
            else:
                self.FFT.FFT_plot(self.path)

    def execute_Sklearn_NN(self):
        if self.path == '':
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка!")
            msg.setText("Файл для анализа не выбран")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()
        else:
            print(self.path)
            if self.flag:
                emotion = self.Sklearn_NN.action(self.path[0])
            else:
                emotion = self.Sklearn_NN.action(self.path)
        self.label8.setText(emotion[0])

    def execute_Keras_NN(self):
        if self.path == '':
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка!")
            msg.setText("Файл для анализа не выбран")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()
        else:
            self.label9.setText("Подождите")
            print(self.path)
            if self.flag:
                emotion = self.Keras_NN.keras_action(self.path[0])
            else:
                emotion = self.Keras_NN.keras_action(self.path)
        self.label9.setText(emotion)

    def execute_F0(self):
        if self.path == '':
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка!")
            msg.setText("Файл для анализа не выбран")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()
            print(self.path)
            print(self.praat_path)
        else:
            print(self.path)
            if self.flag:
                self.F0_Analyser.f0_analysis(self.path[0], self.praat_path)
            else:
                self.F0_Analyser.f0_analysis(self.path, self.praat_path)

    def int_or_str(self, text):
        try:
            return int(text)
        except ValueError:
            return text

    #Визывается отдельно для каждого аудио блока
    def callback(self, indata, frames, time, status):
        self.q.put(indata.copy())

    def speech_recording(self): #
        try:
            if self.args.samplerate is None:
                device_info = sd.query_devices(self.args.device, 'input')
                self.args.samplerate = int(device_info['default_samplerate'])
            if self.args.filename is None:
                    self.args.filename = tempfile.mktemp(prefix='Audio/delme_rec_unlimited_',
                                                    suffix='.wav', dir='')
                    self.path = self.args.filename
                    print(self.path)

            # Проверка на то, открыт ли файл, прежде чем записывать что-либо
            with sf.SoundFile(self.args.filename, mode='x', samplerate=self.args.samplerate,
                              channels=self.args.channels, subtype=self.args.subtype) as file:
                with sd.InputStream(samplerate=self.args.samplerate, device=self.args.device,
                                    channels=self.args.channels, callback=self.callback):
                    n = 0
                    while n < 200:
                        file.write(self.q.get())
                        n = n + 1
                        print(n)
                    self.label.setText("Запись завершена!")
            self.flag = False
        except Exception as e:
            self.parser.exit(type(e).__name__ + ': ' + str(e))


#Граф интерфейс
def application():
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    application()