import random
import os
from re import A,T
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
from scipy import signal
import pywt
import argparse
import array
import math
import wave
import cloudconvert
from signal import Signals
from statistics import mean
import sys
from scipy.signal import find_peaks
from urllib.parse import _ParseResultBase
from pydub import AudioSegment


app = Flask(__name__)

mfccs = 0.0


def extract_data(file_name):
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
    feature = np.array(mfccs).reshape([-1, 1])
    print(mfccs)
    return feature


def spectogram(file_name, user_uid):
    audio, sr = librosa.load(file_name)
    # Plot
    FRAME_SIZE = 2048
    HOP_SIZE = 512
    S_scale = librosa.stft(audio, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    # plot Spectrogram
    plot.figure(figsize=(25, 10))

    def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
        librosa.display.specshow(Y,
                                sr=sr,
                                hop_length=hop_length,
                                x_axis="time",
                                y_axis=y_axis,
                                cmap='viridis')
        plot.colorbar(format="%+2.f")
        plot.savefig(file_name.replace('.wav', '')+'.png', transparent=True)
    Y_scale = np.abs(S_scale) ** 2
    Y_log_scale = librosa.power_to_db(Y_scale)
    plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")

    # firebase
    picture = (file_name.replace('.wav', '')+'.png')
    storage.child(picture).put(picture)
    url = storage.child(picture).get_url(user_uid)
    doccument = db.collection('result').document(user_uid).collection(
        "user_result").document(file_name.replace('.wav', ''))
    doccument.update({'spectogram': url})

def convert(url):
    api = cloudconvert.configure(api_key='eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIxIiwianRpIjoiYmRiNDJmMWQ5YzQyNzYzNWMxODY3Y2ZiYjkyNjQwOWVkNjQyNjljZTNlOGY1NTQ1ODBjYjE3NTMwNzY5MTQ4YzFlMDc3ZWZiZWY3OTQyZGIiLCJpYXQiOjE2NTQ3NDk1NTEuNzQxOTM5LCJuYmYiOjE2NTQ3NDk1NTEuNzQxOTQxLCJleHAiOjQ4MTA0MjMxNTEuNzM1NTQ3LCJzdWIiOiI1NTk2NDUyNiIsInNjb3BlcyI6WyJ1c2VyLndyaXRlIiwidXNlci5yZWFkIiwidGFzay5yZWFkIiwidGFzay53cml0ZSIsIndlYmhvb2sucmVhZCIsIndlYmhvb2sud3JpdGUiLCJwcmVzZXQucmVhZCIsInByZXNldC53cml0ZSJdfQ.cFYOIuJLDNypMGga-yVBSrpLHu85Xq7pHBguPy9bm5W4Fi-ylYTU-bWBo1QuSSC-_z-4hq_rd_bz0D7V5GCLnypnQ-DXgEYopfnWRUueVBre78Qtz1K2oHKbfYxOj4L0aPJ1CaDl_0DJtWkAS-0jgpz5L5V0-Sd7KxxlghcllNQIFSz6x4fufnXCNIS0eoD6tZ7jEbjrRhVCY72hZ-kvElUwUBeCeoLrb1pQHTmzdJo1AUJlogqoygczVVvUFvPD9IyDmvBuVlPG2WeoiE2kDbmFaPqi2Vu4VHLWHPeU275CHL8JrsHzEFgfnJI0LMUpaJYdYA-kJZQpMTox1QkHVcult6e8lF_OYb7nrPLBIHphCFvnHL5ufFF2Z63OVdUuKS30l167MlvKUpY0uGcmIPVLZj76B2AMv4URAfYLm_CE7V9C6Rc0WUa8nemBljnR48IdV6NTjLuG5N3xX4fzVEEJUV8nv90vhwzb5iPIjR2zSRKp2k_2iCuADL9Rs-YNO_-t7rgCTknrkClzOL_78Sf5RkYXflBMJbAyGadTZ4nS-a-Jmzgs3SCyDFpVLOfXX2Wwys3lQP0URm_QD7z12qeBWXShnR2t_fvk3ZPcRFbkjyVfNZkOUnF8a9xay0BR2I8WNw-16NljQVByi1dUMNoEEvNvRcoD2O6OBPrvXM0')
    job = cloudconvert.Job.create(payload={
        "tasks": {
            'import-my-file': {
                'operation': 'import/url',
                'url': url
            },
            'convert-my-file': {
                'operation': 'convert',
                'input': 'import-my-file',
                'output_format': 'wav',
            },
            'export-my-file': {
                'operation': 'export/url',
                'input': 'convert-my-file'
            }
        }
    })

    job = cloudconvert.Job.wait(id=job['id'])
    task=job['tasks']
    tasks = task[0]

    exported_url_task_id = tasks['id']
    res = cloudconvert.Task.wait(id=exported_url_task_id) # Wait for job completion
    file = res.get("result").get("files")[0]
    res = cloudconvert.download(filename=file['filename'], url=file['url'])
    print(res)
    print(file)
    return(res)

def countbpm(filename, user_uid):
    audioname = filename

    spf = wave.open(audioname, "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, dtype="Int16")
    fs = spf.getframerate()

    Time = np.linspace(0, len(signal) / fs, num=len(signal))

    #Find peaks
    peaks = find_peaks(signal, height = 10000, threshold = 1000, distance = 1000)
    height = peaks[1]['peak_heights'] #list of the heights of the peaks
    peak_pos = Time[peaks[0]] #list of the peaks positions

    #Finding the minima
    y2 = signal*-1
    minima = find_peaks(y2)
    min_pos = Time[minima[0]] #list of the minima positions
    min_height = y2[minima[0]] #list of the mirrored minima heights
    # print(signal)
    i = 0
    """ Status = True;
    lowP=[] 
    for signals in signal:
        if(signals < 100 and Status == True and signals > 0):
            i = i+1
            print(signals)
            Status = False
            lowP.append(signals)
        elif(signals > 15000):
            Status = True
    TimelowP = np.linspace(0, len(lowP) / fs, num=len(lowP))
    """
    signalAVG =[]
    signalsSUM = 0

    for signals in signal:
        if(i < 332):
            signalsSUM = signalsSUM + signals
        elif(i >= 332):
            signalAVG.append(signalsSUM/333)
            i = 0
            signalsSUM = 0
        i = i+1
    TimeAVG = np.linspace(0, len(signalAVG) / fs, num=len(signalAVG))
    # print(signalAVG)
    # print(mean(signalAVG))
    print(len(signalAVG))

    maxSignalAVG = int(max(signalAVG)/4)
    print(maxSignalAVG)
    #Find peaks
    peaksAVG = find_peaks(signalAVG, height = maxSignalAVG, threshold = 10, distance = 10)
    heightAVG = peaksAVG[1]['peak_heights'] #list of the heights of the peaks
    peak_posAVG = TimeAVG[peaksAVG[0]] #list of the peaks positions
    startbpm = ((len(peak_posAVG))/2)*6
    x, sr = librosa.load(audioname) 
    ipd.Audio(x, rate=sr)
    print(x)
    tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=startbpm, units='time')
    print(tempo)
    print(beat_times)
    doccument = db.collection('result').document(user_uid).collection(
        "user_result").document(filename.replace('.wav', ''))
    doccument.update({'bpm': tempo})
    return(tempo)

filelist = [f for f in os.listdir(".") if f.endswith(
    ".mp3") or f.endswith(".wav") or f.endswith(".png")]
for f in filelist:
    os.remove(os.path.join(".", f))

config = {
    "apiKey": "AIzaSyAmrjkU6rxiZoAHnK4ylL4JPqFdZEWP-kw",
    "authDomain": "mypros-283015.firebaseapp.com",
    "databaseURL": "https://MYPROS.firebaseio.com",
    "projectId": "mypros-283015",
    "storageBucket": "mypros-283015.appspot.com",
    "serviceAccount": "serviceAccountKey.json"
}
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
firebase_storage = pyrebase.initialize_app(config)

db = firestore.client()
storage = firebase_storage.storage()
auth = firebase_storage.auth()
email = "noppawitearth123456@gmail.com"
password = "123456"
user = auth.sign_in_with_email_and_password(email, password)
af = load_model('af_fa.h5')
brady = load_model('brady.h5')
murmur = load_model('new_murmur.h5')

app = Flask(__name__)


@app.route("/text", methods=["GET"])
def text():
    name_file = request.values["name"]
    return('hello' + name_file)


@app.route("/heart", methods=["GET"])
def audio():
    data = []
    user_uid = request.values["uid"]
    url = request.values["url"]
    print(user_uid)
    h_sound = convert(url)
    test1 = extract_data(h_sound)
    spectogram(h_sound, user_uid)
    bpm = countbpm(h_sound, user_uid)
    data.append(test1)
    af_result = af.predict(np.array(data))
    brady_result = brady.predict(np.array(data))
    murmur_result = murmur.predict(np.array(data))
    y = af_result[0]
    a = brady_result[0]
    b = murmur_result[0]
    af_return = y[0]*100
    brady_return = a[0]*100
    murmur_return = b[0]*100
    returnvalue = [af_return, brady_return, murmur_return, bpm]
    return(str(returnvalue))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
