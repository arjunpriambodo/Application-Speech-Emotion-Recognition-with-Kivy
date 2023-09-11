import librosa
import soundfile
import numpy as np 
from sklearn.model_selection import train_test_split 
import tensorflow as tf

import pyaudio
import wave

from kivy.app import App

from kivy.core.window import Window
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = ["./assets/audio/output.wav", "./assets/audio/output2.wav", "./assets/audio/output3.wav", "./assets/audio/output4.wav", 
                        "./assets/audio/output5.wav", "./assets/audio/output6.wav", "./assets/audio/output7.wav", "./assets/audio/output8.wav"]

def extract_feature(file_name):    
    with soundfile.SoundFile(file_name) as sound_file:        
        X = sound_file.read(dtype="float32")        
        sample_rate=sound_file.samplerate                  
        stft=np.abs(librosa.stft(X))
        mfcc=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)    
        mfcc_std = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        chroma_std = np.std(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)            
        mel_std = np.std(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        contrast_std = np.std(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        tonnetz_std = np.std(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return np.hstack((mfcc, chroma, mel, contrast, tonnetz,
           mfcc_std, chroma_std, mel_std, contrast_std, tonnetz_std))

X = np.load('./extracted/full/x.npy')
y = np.load('./extracted/full/y.npy')

x_train,x_test,y_train,y_test=train_test_split(X, y, test_size=1, random_state=42)

def model_dense():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(axis=-1, input_shape=(x_train.shape[1:])))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(16, activation='softmax'))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    return model

model = model_dense()
print(model.summary())

tf.keras.utils.plot_model(model,to_file='mlp_model.pdf',show_shapes=True)

hist = model.fit(x_train,
                 y_train,
                 epochs=1,
                shuffle=True,
                 
                validation_split=0.2, 
                batch_size=16)

evaluate = model.evaluate(x_train, y_train, batch_size=16)

def record_and_predict(self):
    print("* recording")
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    while True:
        for i in range(8):
            print("Recording " + str(i+1) + " of 8")
            frames = []
            for j in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            print("Finished recording " + str(i+1) + " of 8")
            waveFile = wave.open(WAVE_OUTPUT_FILENAME[i], 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(p.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()
            
        if i == 7:
            break   

    hasil = []
    for i in range(1):
        if i == 0: 
            f = extract_feature('./assets/audio/output.wav').reshape(1, -1)
            pred_emotion = model.predict(f, batch_size=16)[0]
            result = "{:.2f}%".format(pred_emotion[0]*100)
            hasil.append(result)
        if i == 0:
            f = extract_feature('./assets/audio/output2.wav').reshape(1, -1)
            pred_emotion = model.predict(f, batch_size=16)[0]
            result = "{:.2f}%".format(pred_emotion[1]*100)
            hasil.append(result)
        if i == 0:
            f = extract_feature('./assets/audio/output3.wav').reshape(1, -1)
            pred_emotion = model.predict(f, batch_size=16)[0]
            result = "{:.2f}%".format(pred_emotion[2]*100)
            hasil.append(result)
        if i == 0:
            f = extract_feature('./assets/audio/output4.wav').reshape(1, -1)
            pred_emotion = model.predict(f, batch_size=16)[0]
            result = "{:.2f}%".format(pred_emotion[3]*100)
            hasil.append(result)
        if i == 0:
            f = extract_feature('./assets/audio/output5.wav').reshape(1, -1)
            pred_emotion = model.predict(f, batch_size=16)[0]
            result = "{:.2f}%".format(pred_emotion[4]*100)
            hasil.append(result)
        if i == 0:
            f = extract_feature('./assets/audio/output6.wav').reshape(1, -1)
            pred_emotion = model.predict(f, batch_size=16)[0]
            result = "{:.2f}%".format(pred_emotion[5]*100)
            hasil.append(result)
        if i == 0:
            f = extract_feature('./assets/audio/output7.wav').reshape(1, -1)
            pred_emotion = model.predict(f, batch_size=16)[0]
            result = "{:.2f}%".format(pred_emotion[6]*100)
            hasil.append(result)
        if i == 0:
            f = extract_feature('./assets/audio/output8.wav').reshape(1, -1)
            pred_emotion = model.predict(f, batch_size=16)[0]
            result = "{:.2f}%".format(pred_emotion[7]*100)
            hasil.append(result)
        if i == 1:
            break

        return(hasil)

    
class SER(App):
    def build(self):
        self.window = RelativeLayout()
        self.window.cols = 1
        Window.clearcolor = (1, 1, 1, 1)

        # image widget
        self.window.add_widget(Image(source="./assets/image/logo.jpg",
                                 size_hint = (0.3, 0.2),
                                 pos_hint = {"center_x": 0.5, "center_y": 0.85}))
        
        self.emosi = Label(
                        text="Ucapkan Dengan Emosi: \n Natural, Kalem, Senang, Sedih, Marah, Tenang, Jijik, Terkejut",
                        font_size=14,
                        pos_hint= {'center_x': 0.5,'center_y': 0.65},
                        halign="center",
                        color='#000000')
        self.window.add_widget(self.emosi)

        # label widget
        self.greeting = Label(
                        text= "Speech Emotion Recognition",
                        pos_hint= {'center_x': 0.5,'center_y': 0.4},
                        font_size= 18,
                        bold= True,
                        italic= True,
                        color= '#000000'
                        )
        self.window.add_widget(self.greeting)

        # button widget
        self.button = Button(
                      text= "Tekan Untuk Merekam",
                      size_hint= (0.3,0.1),
                      pos_hint= {'center_x': 0.5,'center_y': 0.1},
                      bold= True,
                      background_color ='#000000',
                      )
        self.button.bind(on_press=self.callback, on_release=self.releaseback)
        self.window.add_widget(self.button)

        return self.window
    def releaseback(self, instance):
        self.emosi.text = "Emosi Yang Terdeteksi:"
        self.button.text = "Tekan Untuk Merekam"
        self.emot = self.find(self)
        self.greeting.text = "Natural = " + self.emot[0] + " \n " + "Kalem = " + self.emot[1] + " \n " + "Senang = " + self.emot[2] + " \n " + "Sedih = " + self.emot[3] + " \n " + "Marah = " + self.emot[4] + " \n " + "Tenang = " + self.emot[5] + " \n " + "Jijik = " + self.emot[6] + " \n " + "Terkejut = " + self.emot[7]

    def find(self, instance):
        self.pred_emotion = record_and_predict(self)
        return self.pred_emotion
    
    def callback(self, instance):
        self.button.text = "Recording..."
        self.greeting.text = "Beralih emosi setiap 4 detik!!!"

        
if __name__ == "__main__":
    SER().run()