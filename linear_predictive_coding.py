import librosa as lbr
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import IPython.display as ipd
from IPython.display import display


class LPC():
    #def __init__(self):
    #    self.pa = "delme_rec_unlimited_3u71xz4b.wav"
    #    self.LPC_plot(self.pa)

    def LPC_plot(self, path):
        # Примаем речь на вход
        print(path)
        self.x, self.samplerate = lbr.load(path, duration=5, sr=32000)
        self.x/=np.abs(self.x).max()
        print('Audio Length:',np.size(self.x))
        display(ipd.Audio(self.x, rate = self.samplerate ))

        self.L=10 # Длина предсказания
        self.len0 = np.max(np.size(self.x)) # Запизнули длину аудио в переменную
        self.e = np.zeros(np.size(self.x)) # Инициализация переменной ошибки предсказания
        self.blocks = np.int(np.floor(self.len0/640)) # Итоговое количество блоков, на который будет разбит аудиофайл
        self.state = np.zeros(self.L) # Состояние памяти фильтра предсказания
        # Строим матрицу А из блоков, длиной 640 и обрабатыавем:
        self.h=np.zeros((self.blocks,self.L)) # Память коэффециэнта предсказания


        for m in range(0,self.blocks):
            A = np.zeros((640-self.L,self.L)) # Хитрый ход: до 630, чтобы не было нулей в матрице
            for n in range(0,640-self.L):
                A[n,:] = np.flipud(self.x[m*640+n+np.arange(self.L)])
        
            # Построим наш желаемый целевой сигнал d на один отсчет в будущее:
            d=self.x[m*640+np.arange(self.L,640)];
            # Вычислим фильтр предсказания:
            self.h[m,:] = np.dot(np.dot(np.linalg.pinv(np.dot(A.transpose(),A)), A.transpose()), d)
            hperr = np.hstack([1, -self.h[m,:]])
            self.e[m*640+np.arange(0,640)], self.state = sp.lfilter(hperr,[1],self.x[m*640+np.arange(0,640)], zi=self.state)
        
        #Теперь среднеквадратичная ошибка равна:
        print ("Средняя квадратичная ошибка:", np.dot(self.e.transpose(),self.e)/np.max(np.size(self.e)))
        #The average squared error is: 0.000113347859337
        #Мы видим, что это всего лишь 1/4 от предыдущей ошибки предсказания!
        print ("Сравните это со среднеквадратичной мощностью сигнала:", np.dot(self.x.transpose(),self.x)/np.max(np.size(self.x)))
        #0.00697569381701
        print ("Отношение сигнал / ошибка:", np.dot(self.x.transpose(),self.x)/np.dot(self.e.transpose(),self.e))
        #61.5423516403
        #Таким образом, наша энергия предопределения LPC более чем в 61 раз меньше, чем энергия сигнала
        #Читаем ошибку предсказания
        display(ipd.Audio(self.e, rate = self.samplerate ))
        
        #Взглянем на сигнал и его ошибку прогноза:
        plt.figure(figsize=(10,8))
        plt.plot(self.x)
        #plt.hold(True)
        plt.plot(self.e,'r')
        plt.xlabel('Sample')
        plt.ylabel('Normalized Value')
        plt.legend(('Original','Prediction Error'))
        plt.title('LPC Coding')
        plt.grid()
        plt.show()
        
        #Decoder:
        xrek=np.zeros(self.x.shape) #initialize reconstructed signal memory
        self.state = np.zeros(self.L) #Initialize Memory state of prediction filter
        for m in range(0,self.blocks):
            hperr = np.hstack([1, -self.h[m,:]])
            #predictive reconstruction filter: hperr from numerator to denominator:
            xrek[m*640+np.arange(0,640)] , self.state = sp.lfilter([1], hperr,self.e[m*640+np.arange(0,640)], zi=self.state)
        
        #Listen to the reconstructed signal:
        display(ipd.Audio(xrek, rate = self.samplerate ))

if __name__ == '__main__':
    LPC()