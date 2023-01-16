import random
import os
from re import A
import numpy as np
import librosa
import librosa.display
import pyrebase
import matplotlib.pyplot as plot
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from pydub import AudioSegment
from scipy.io import wavfile
import IPython.display as ipd
import wavio
import firebase_admin
from firebase_admin import credentials, firestore
import argparse
import array
import math
import wave
import cloudconvert
import requests
import numpy as np
from matplotlib.pyplot import figure
from scipy.signal import savgol_filter, find_peaks,butter,hilbert, filtfilt
import sys
import contextlib
import timeit
from threading import Thread
import multiprocessing as mp

app = Flask(__name__)

mfccs = 0.0


def extract_data(file_name):
    print('extract start!')
    # function to load files and extract features
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # we extract mfcc feature from data
        global mfccs
        mfccs = np.mean(librosa.feature.mfcc(
            y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ")
    features = np.array(mfccs).reshape([-1, 1])
    print(mfccs)
    return features

def heart_rate(filename):
    print('heartrate start!')
    y, fs = librosa.load(filename, sr=40000)
    def homomorphic_envelope(y, fs, f_LPF=8, order=3):
        b, a = butter(order, 2 * f_LPF / fs, 'low')
        he = np.exp(filtfilt(b, a, np.log(np.abs(hilbert(y)))))
        return he
    he = homomorphic_envelope(y, fs)
    x = he - np.mean(he)
    corr = np.correlate(x, x, mode='same')
    corr = corr[int(corr.size/2):]
    min_index = int(0.5*fs)
    max_index = int(2*fs)
    index = np.argmax(corr[min_index:max_index])
    true_index = index+min_index
    heartRate = 60/(true_index/fs)
    plot.plot(y)
    plot.plot(he, linewidth=2)
    plot.savefig('heart graph.png')
    return heartRate


config = {
    "apiKey": "AIzaSyAmrjkU6rxiZoAHnK4ylL4JPqFdZEWP-kw",
    "authDomain": "mypros-283015.firebaseapp.com",
    "databaseURL": "https://MYPROS.firebaseio.com",
    "projectId": "mypros-283015",
    "storageBucket": "mypros-283015.appspot.com",
    "serviceAccount": "serviceAccountKey.json"
}
cred = credentials.Certificate("mypros-283015-firebase-adminsdk-eewnd-fa5c258fcf.json")
firebase_admin.initialize_app(cred)
firebase_storage = pyrebase.initialize_app(config)

db = firestore.client()
storage = firebase_storage.storage()

af = load_model('mir.h5')
murmur = load_model('murmur.h5')

app = Flask(__name__)


@app.route("/text", methods=["POST"])
def text():
    name_file = request.values["name"]
    return('hello' + name_file)


@app.route("/heart", methods=["POST"])
def audio():
    data = []
    url = request.values["url"]
    start = timeit.default_timer()
    #adjust filename
    ch = '='
    listOfWords = url.split(ch, 1)
    if len(listOfWords) > 0: 
        url_name = listOfWords[1]
    listOfWords = url.split(ch, 1)
    if len(listOfWords) > 0: 
        url_name = listOfWords[1]
    filename = url_name+".mp3"
    print(filename)
    #download files
    response = requests.get(url)
    open(filename, "wb").write(response.content)
    #Processing
    pool = mp.Pool(processes=1)
    rate = pool.apply_async(heart_rate, (filename,))
    extract = pool.apply_async(extract_data, (filename,))
    features = extract.get()
    heartRate = rate.get()
    #AI stuff
    pool.close()
    pool.join()
    data.append(features)
    af_result = af.predict(np.array(data))
    murmur_result = murmur.predict(np.array(data))
    y = af_result[0]
    b = murmur_result[0]
    af_return = y[0]*100
    murmur_return = b[0]*100
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    returnvalue = [af_return, murmur_return,heartRate]
    print('Retrun: ', returnvalue)
    return(str(returnvalue))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
