import librosa as lbr
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import IPython.display as ipd
from IPython.display import display


class LPC():
    def LPC_plot(self, path):
        # Примаем речь на вход
        print(path)
        x, samplerate = lbr.load(path, duration=5, sr=32000)
        x/=np.abs(x).max()
        print('Audio Length:',np.size(x))
        display(ipd.Audio(x, rate = samplerate ))
        L=10 # Длина предсказания
        len0 = np.max(np.size(x)) # Запизнули длину аудио в переменную
        e = np.zeros(np.size(x)) # Инициализация переменной ошибки предсказания
        blocks = np.int(np.floor(len0/640)) # Итоговое количество блоков, на который будет разбит аудиофайл
        state = np.zeros(L) # Состояние памяти фильтра предсказания
        # Строим матрицу А из блоков, длиной 640 и обрабатыавем:
        h=np.zeros((blocks,L)) # Память коэффециэнта предсказания

        for m in range(0,blocks):
            A = np.zeros((640-L,L)) # Хитрый ход: до 630, чтобы не было нулей в матрице
            for n in range(0,640-L):
                A[n,:] = np.flipud(x[m*640+n+np.arange(L)])
        
            # Построим наш желаемый целевой сигнал d на один отсчет в будущее:
            d=x[m*640+np.arange(L,640)];
            # Вычислим фильтр предсказания:
            h[m,:] = np.dot(np.dot(np.linalg.pinv(np.dot(A.transpose(),A)), A.transpose()), d)
            hperr = np.hstack([1, -h[m,:]])
            e[m*640+np.arange(0,640)], state = sp.lfilter(hperr,[1],x[m*640+np.arange(0,640)], zi=state)
        
        #Теперь среднеквадратичная ошибка равна:
        print ("Средняя квадратичная ошибка:", np.dot(e.transpose(),e)/np.max(np.size(e)))
        #The average squared error is: 0.000113347859337
        #Мы видим, что это всего лишь 1/4 от предыдущей ошибки предсказания!
        print ("Сравните это со среднеквадратичной мощностью сигнала:", np.dot(x.transpose(),x)/np.max(np.size(x)))
        #0.00697569381701
        print ("Отношение сигнал / ошибка:", np.dot(x.transpose(),x)/np.dot(e.transpose(),e))
        #61.5423516403
        #Таким образом, наша энергия предопределения LPC более чем в 61 раз меньше, чем энергия сигнала
        #Читаем ошибку предсказания
        display(ipd.Audio(e, rate = samplerate ))
        
        #Взглянем на сигнал и его ошибку прогноза:
        plt.figure(figsize=(10,8))
        plt.plot(x)
        #plt.hold(True)
        plt.plot(e,'r')
        plt.xlabel('Sample')
        plt.ylabel('Normalized Value')
        plt.legend(('Original','Prediction Error'))
        plt.title('LPC Coding')
        plt.grid()
        plt.show()
        
        #Decoder:
        xrek=np.zeros(x.shape) #initialize reconstructed signal memory
        state = np.zeros(L) #Initialize Memory state of prediction filter
        for m in range(0,blocks):
            hperr = np.hstack([1, -h[m,:]])
            #predictive reconstruction filter: hperr from numerator to denominator:
            xrek[m*640+np.arange(0,640)] , state = sp.lfilter([1], hperr,e[m*640+np.arange(0,640)], zi=state)
        
        #Listen to the reconstructed signal:
        display(ipd.Audio(xrek, rate = samplerate ))

if __name__ == '__main__':
    LPC()