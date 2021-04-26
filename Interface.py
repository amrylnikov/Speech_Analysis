from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QPushButton, QMessageBox
import sys
from streaming_sound import AudioStream
from pydub import AudioSegment
from file_analysis import Analysis
from linear_predictive_coding import LPC
from Mel_Frequency_Cepstral_Coefficients import MFCC
from file_fft_analysis import FFT
from emotion_analysis import NN
from f0_analysis import F0_Analizer
import sounddevice as sd
import soundfile as sf
import argparse
import tempfile
import queue

class Window(QMainWindow): #класс-наследник от главного окна
    def __init__(self): #функция при создании
        super(Window, self).__init__() #вызываем конструктор из родительского класса
        self.a = "3"
        self.path = ''
        self.praat_path = 'Praat/Praat_for_f0.txt'
        self.flag = True
        #self.Window2 = Window2()
        self.Analysis = Analysis()
        self.LPC = LPC()
        self.MFCC = MFCC()
        self.FFT = FFT()
        self.NN = NN()
        self.F0_Analizer = F0_Analizer()

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
        self.setWindowTitle("Анализ речи") # название окна
        self.setGeometry(50, 50, 530, 500) #насколько окно отодвинется от левого верхнего угра + ширина и высота самого окна

        self.new_text = QtWidgets.QLabel(self) #создаём текстовую переменную, чтобы потом её менять

        self.label = QtWidgets.QLabel(self)
        self.label.move(280, 55)
        self.label.setText("Запись голоса будет идти 5 секунд")
        self.label.adjustSize()

        self.label2 = QtWidgets.QLabel(self)
        self.label2.move(280, 105)
        self.label2.setText("Фотмат файла: .wav")
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
        self.label5.setText("Linear Prediction Coding")
        self.label5.adjustSize()

        self.label6 = QtWidgets.QLabel(self)
        self.label6.move(280, 305)
        self.label6.setText("Mel-Frequency_Cepstral_Coefficients")
        self.label6.adjustSize()

        self.label7 = QtWidgets.QLabel(self)
        self.label7.move(280, 355)
        self.label7.setText("Fast Fourier transform")
        self.label7.adjustSize()

        self.label8 = QtWidgets.QLabel(self)
        self.label8.move(280, 405)
        self.label8.setText("*Здесь будет указан ответ*")
        self.label8.adjustSize()

        self.label9 = QtWidgets.QLabel(self)
        self.label9.move(280, 455)
        self.label9.setText("Используются 8 методов анализа")
        self.label9.adjustSize()

        self.btn = QtWidgets.QPushButton(self) #создали кнопку
        self.btn.move(50,50) #установили место
        self.btn.setText("Запись речи")
        self.btn.setFixedWidth(200) #фиксируем ширину для кнопки
        self.btn.clicked.connect(self.SpeechRecording) #Вызывает функцию при нажатии

        self.btn2 = QtWidgets.QPushButton(self)  # создали кнопку
        self.btn2.move(50, 100)  # установили место
        self.btn2.setText("Открыть файл")
        self.btn2.setFixedWidth(200)  # фиксируем ширину для кнопки
        self.btn2.clicked.connect(self.get_path)  # Вызывает функцию при нажатии

        self.btn3 = QtWidgets.QPushButton(self) #создали кнопку
        self.btn3.move(50,150) #установили место
        self.btn3.setText("Анализ в реальном времени")
        self.btn3.setFixedWidth(200) #фиксируем ширину для кнопки
        self.btn3.clicked.connect(AudioStream) #Вызывает функцию при нажатии

        self.btn4 = QtWidgets.QPushButton(self)  # создали кнопку
        self.btn4.move(50, 200)  # установили место
        self.btn4.setText("Запустить обычный анализ")
        self.btn4.setFixedWidth(200)  # фиксируем ширину для кнопки
        self.btn4.clicked.connect(self.execute_file_analysis)  # Вызывает функцию при нажатии

        self.btn5 = QtWidgets.QPushButton(self)  # создали кнопку
        self.btn5.move(50, 250)  # установили место
        self.btn5.setText("Запустить анализ LPC")
        self.btn5.setFixedWidth(200)  # фиксируем ширину для кнопки
        self.btn5.clicked.connect(self.execute_LPC)  # Вызывает функцию при нажатии

        self.btn6 = QtWidgets.QPushButton(self)  # создали кнопку
        self.btn6.move(50, 300)  # установили место
        self.btn6.setText("Запустить анализ MFCC")
        self.btn6.setFixedWidth(200)  # фиксируем ширину для кнопки
        self.btn6.clicked.connect(self.execute_MFCC)  # Вызывает функцию при нажатии

        self.btn7 = QtWidgets.QPushButton(self)  # создали кнопку
        self.btn7.move(50, 350)  # установили место
        self.btn7.setText("Запустить анализ FFT")
        self.btn7.setFixedWidth(200)  # фиксируем ширину для кнопки
        self.btn7.clicked.connect(self.execute_FFT)  # Вызывает функцию при нажатии

        self.btn8 = QtWidgets.QPushButton(self)  # создали кнопку
        self.btn8.move(50, 400)  # установили место
        self.btn8.setText("Характеристика речи:")
        self.btn8.setFixedWidth(200)  # фиксируем ширину для кнопки
        self.btn8.clicked.connect(self.execute_NN)  # Вызывает функцию при нажатии

        self.btn9 = QtWidgets.QPushButton(self)  # создали кнопку
        self.btn9.move(50, 450)  # установили место
        self.btn9.setText("Анализ F0")
        self.btn9.setFixedWidth(200)  # фиксируем ширину для кнопки
        self.btn9.clicked.connect(self.execute_F0)  # Вызывает функцию при нажатии


    # просьба открыть файл
    def get_path(self):
        self.flag = True
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.path = QFileDialog.getOpenFileName(self,"Select Audio File", "","All Files (*);;Audio File (*.wav)", options=options)
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

            #self.Window2.input1.setText(self.a)
            #self.Window2.input2.setText(str(self.path))
            #self.Window2.displayInfo()

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

    def execute_NN(self):
        if self.path == '':
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка!")
            msg.setText("Файл для анализа не выбран")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()
        else:
            print(self.path)
            if self.flag:
                emotion = self.NN.action(self.path[0])
            else:
                emotion = self.FFT.FFT_plot(self.path)
        self.label8.setText(emotion[0])

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
                self.F0_Analizer.f0_analysis(self.path[0], self.praat_path)
            else:
                self.F0_Analizer.f0_analysis(self.path, self.praat_path)

    #Функция помогает в передаче аргументов
    def int_or_str(self, text):
        try:
            return int(text)
        except ValueError:
            return text

    #Визывается отдельно для каждого аудио блока
    def callback(self, indata, frames, time, status):
        self.q.put(indata.copy())

    def SpeechRecording(self): #TODO добавить логическую переменную, после первой записи и выводить диалоговое окно если она не работает
        try:
            if self.args.samplerate is None:
                device_info = sd.query_devices(self.args.device, 'input')
                # soundfile expects an int, sounddevice provides a float:
                self.args.samplerate = int(device_info['default_samplerate'])
            if self.args.filename is None:
                    self.args.filename = tempfile.mktemp(prefix='delme_rec_unlimited_',
                                                    suffix='.wav', dir='')
                    self.path = self.args.filename
                    print(self.path)

            # Проперка на то, открыт ли файл, прежде чем записывать что-либо
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
    app = QApplication(sys.argv) #Создаём приложение в целом и запихиваем туда данные компа
    window = Window()# Создаём главное окно, указывая нужный класс

    window.show() #выводим окно
    sys.exit(app.exec_()) #обеспечиваем выход из окна

if __name__ == "__main__": #Запустили ли мы этот файл как основной
    application()