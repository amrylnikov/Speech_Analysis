from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QWidget, QPushButton, QMessageBox, QMdiSubWindow, QFrame
import sys
from streaming_sound import *
from streaming_sound_puqt import AudioStream
from app import SpeechRecognition
from w2 import *
from pydub import AudioSegment

class Window(QMainWindow): #класс-наследник от главного окна
    def __init__(self): #функция при создании
        super(Window, self).__init__() #вызываем конструктор из родительского класса
        self.a = "3"
        self.path = ''
        self.Window2 = Window2()
        self.Window1 = Window1()

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

        # Кнопка "Распознавание речи"
        self.btn = QtWidgets.QPushButton(self) #создали кнопку
        self.btn.move(50,50) #установили место
        self.btn.setText("Запись речи")
        self.btn.setFixedWidth(200) #фиксируем ширину для кнопки
        self.btn.clicked.connect(self.hereWeGo) #Вызывает функцию при нажатии (было - SpeechRecognition)

        #Текстовое поле рядом с кнопкой
        self.label = QtWidgets.QLabel(self)
        self.label.move(280, 55)
        self.label.setText("Запись голоса будет идти 5 секунд")
        self.label.adjustSize()

        #Текстовое поле рядом с кнопкой
        self.label2 = QtWidgets.QLabel(self)
        self.label2.move(280, 105)
        self.label2.setText("Фотмат файла: .wav")
        self.label2.adjustSize()

        #Текстовое поле рядом с кнопкой
        self.label3 = QtWidgets.QLabel(self)
        self.label3.move(280, 155)
        self.label3.setText("Визуализация громкости и частоты")
        self.label3.adjustSize()

        #Текстовое поле рядом с кнопкой
        self.label4 = QtWidgets.QLabel(self)
        self.label4.move(280, 205)
        self.label4.setText("Для анализа необходим файл")
        self.label4.adjustSize()


        # Кнопка "Открыть файл"
        self.btn2 = QtWidgets.QPushButton(self)  # создали кнопку
        self.btn2.move(50, 100)  # установили место
        self.btn2.setText("Открыть файл")
        self.btn2.setFixedWidth(200)  # фиксируем ширину для кнопки
        self.btn2.clicked.connect(self.lame)  # Вызывает функцию при нажатии

        # Кнопка "Вывод спектра"
        self.btn3 = QtWidgets.QPushButton(self) #создали кнопку
        self.btn3.move(50,150) #установили место
        self.btn3.setText("Анализ спектрограммы")
        self.btn3.setFixedWidth(200) #фиксируем ширину для кнопки
        self.btn3.clicked.connect(AudioStrea1m) #Вызывает функцию при нажатии

        # Кнопка "анализ"
        self.btn4 = QtWidgets.QPushButton(self)  # создали кнопку
        self.btn4.move(50, 200)  # установили место
        self.btn4.setText("Запустить анализ")
        self.btn4.setFixedWidth(200)  # фиксируем ширину для кнопки
        self.btn4.clicked.connect(self.show_window_2)  # Вызывает функцию при нажатии

    # просьба открыть файл
    def lame(self):
        self.path = QFileDialog.getOpenFileNames()
        print(self.path)

    def show_window_1(self):
        self.Window1.displayInfo()

    def show_window_2(self):
        print(self.path)
        if self.path == '':
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка!")
            msg.setText("Файл для анализа не выбран")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()
        else:
            self.Window2.input1.setText(self.a)
            self.Window2.input2.setText(str(self.path))
            self.Window2.displayInfo()

    def int_or_str(self, text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def hereWeGo(self):
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

            # Make sure the file is opened before recording anything:
            with sf.SoundFile(self.args.filename, mode='x', samplerate=self.args.samplerate,
                              channels=self.args.channels, subtype=self.args.subtype) as file:
                with sd.InputStream(samplerate=self.args.samplerate, device=self.args.device,
                                    channels=self.args.channels, callback=self.callback):
                    print('#' * 80)
                    print('press Ctrl+C to stop the recording')
                    print('#' * 80)
                    n = 0
                    while n < 200:
                        file.write(self.q.get())
                        n = n + 1
                        print(n)
                    self.label.setText("Запись завершена!")
        except KeyboardInterrupt:
            print('\nRecording finished: ' + repr(self.args.filename))
            self.parser.exit(0)
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