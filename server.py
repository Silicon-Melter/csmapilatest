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
from scipy import signal
import pywt
import argparse
import array
import math
import wave
import cloudconvert

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
        plot.savefig(file_name.replace('.mp3', '')+'.png', transparent=True)
    Y_scale = np.abs(S_scale) ** 2
    Y_log_scale = librosa.power_to_db(Y_scale)
    plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")

    # firebase
    picture = (file_name.replace('.mp3', '')+'.png')
    storage.child(picture).put(picture)
    url = storage.child(picture).get_url('')
    doccument = db.collection('result').document(user_uid).collection(
        "user_result").document(file_name.replace('.mp3', ''))
    doccument.update({'spectogram': url})


def read_wav(filename):
    # open file, get metadata for audio
    try:
        wf = wave.open(filename, "rb")
    except IOError as e:
        print(e)
        return

    # typ = choose_type( wf.getsampwidth() ) # TODO: implement choose_type
    nsamps = wf.getnframes()
    assert nsamps > 0

    fs = wf.getframerate()
    assert fs > 0

    # Read entire file and make into an array
    samps = list(array.array("i", wf.readframes(nsamps)))

    try:
        assert nsamps == len(samps)
    except AssertionError:
        print(nsamps, "not equal to", len(samps))

    return samps, fs


def peak_detect(data):
    max_val = np.amax(abs(data))
    peak_ndx = np.where(data == max_val)
    if len(peak_ndx[0]) == 0:  # if nothing found then the max must be negative
        peak_ndx = np.where(data == -max_val)
    return peak_ndx


def bpm_detector(data, fs):
    cA = []
    cD = []
    correl = []
    cD_sum = []
    levels = 4
    max_decimation = 2 ** (levels - 1)
    min_ndx = math.floor(60.0 / 300 * (fs / max_decimation))
    max_ndx = math.floor(60.0 / 40 * (fs / max_decimation))

    for loop in range(0, levels):
        cD = []
        # 1) DWT
        if loop == 0:
            [cA, cD] = pywt.dwt(data, "db4")
            cD_minlen = len(cD) / max_decimation + 1
            cD_sum = np.zeros(math.floor(cD_minlen))
        else:
            [cA, cD] = pywt.dwt(cA, "db4")

        # 2) Filter
        cD = signal.lfilter([0.01], [1 - 0.99], cD)

        # 4) Subtract out the mean.

        # 5) Decimate for reconstruction later.
        cD = abs(cD[:: (2 ** (levels - loop - 1))])
        cD = cD - np.mean(cD)

        # 6) Recombine the signal before ACF
        #    Essentially, each level the detail coefs (i.e. the HPF values) are concatenated to the beginning of the array
        cD_sum = cD[0: math.floor(cD_minlen)] + cD_sum

    # Adding in the approximate data as well...
    cA = signal.lfilter([0.01], [1 - 0.99], cA)
    cA = abs(cA)
    cA = cA - np.mean(cA)
    cD_sum = cA[0: math.floor(cD_minlen)] + cD_sum

    # ACF
    correl = np.correlate(cD_sum, cD_sum, "full")

    midpoint = math.floor(len(correl) / 2)
    correl_midpoint_tmp = correl[midpoint:]
    peak_ndx = peak_detect(correl_midpoint_tmp[min_ndx:max_ndx])

    peak_ndx_adjusted = peak_ndx[0] + min_ndx
    bpm = 60.0 / peak_ndx_adjusted * (fs / max_decimation)
    print(bpm)
    return bpm, correl


def main_bpm(url,user_uid,file_name):
    filename = convert(url)
    samps, fs = read_wav(filename)
    data = []
    correl = []
    bpm = 0
    n = 0
    nsamps = len(samps)
    window_samps = int(3 * fs)
    samps_ndx = 0  # First sample in window_ndx
    max_window_ndx = math.floor(nsamps / window_samps)
    bpms = np.zeros(max_window_ndx)

    # Iterate through all windows
    for window_ndx in range(0, max_window_ndx):

        # Get a new set of samples
        # print(n,":",len(bpms),":",max_window_ndx_int,":",fs,":",nsamps,":",samps_ndx)
        data = samps[samps_ndx : samps_ndx + window_samps]
        if not ((len(data) % window_samps) == 0):
            raise AssertionError(str(len(data)))

        bpm, correl_temp = bpm_detector(data, fs)
        if bpm is None:
            continue
        bpms[window_ndx] = bpm
        correl = correl_temp

        # Iterate at the end of the loop
        samps_ndx = samps_ndx + window_samps

        # Counter for debug...
        n = n + 1

    bpm = np.median(bpms)
    realbpm = bpm/2
    doccument = db.collection('result').document(user_uid).collection(
        "user_result").document(file_name.replace('.mp3', ''))
    doccument.update({'bpm': realbpm})
    print(bpm/2)
    return(bpm/2)

def convert(url):
    api = cloudconvert.configure(api_key='eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIxIiwianRpIjoiNDZmMDYzNWUyYTRmYmRkZWE3OGQ2Yzc4OTNjYjQyZTBjMTY2ODBmNDc3NGUzMzYyMDEyYjRmNTNlYTFkMTRlODExNGM4YzM1ZjcyYjljNDQiLCJpYXQiOjE2NDU2OTkyNDkuMTEwNDY5LCJuYmYiOjE2NDU2OTkyNDkuMTEwNDcxLCJleHAiOjQ4MDEzNzI4NDkuMTA2MjkzLCJzdWIiOiI1NjQyNjMzNyIsInNjb3BlcyI6WyJ1c2VyLnJlYWQiLCJ1c2VyLndyaXRlIiwidGFzay5yZWFkIiwidGFzay53cml0ZSIsIndlYmhvb2sucmVhZCIsIndlYmhvb2sud3JpdGUiLCJwcmVzZXQucmVhZCIsInByZXNldC53cml0ZSJdfQ.IQUZ47Iwy08mXjfEs4EtACseQ4VCVFFYfOXKg7muRNRqDcdLykU5NlHCyvM74DoBcGx8UDexIh2BkpD7WVCN6Jez5spif7xuTRdRisC-fWODQBkqj_Vz4qbREi_lSa6ai_YUL3X7pHz7xYQIwAo2WzSiEls-N1k405fgQ0-oXAsG4i3ZxyPjeEZrT1Y7_2FptoDk26bArm6ebWe5QV0AHix5frJf6dyj44TBTKsrHiB_rTx_h8_xmyY1fzycfibmCsC2SLQGeuwggaIdQwqZdc3aLALdlnlYoGS9CALQ9seSWn48bJsbvr66PHllWUb3bFkrGo1u4DoTLhF1U03nMy01OmO4pf8fKXhmOMSAjAlzEmFqOPVYf9ZH8zATHO899TLxWhvVKMPrNaGUstqvMyQ0d9nvrvly2w9L1mTE9q6yNBxJGNM5Ptz4s724KNjhsReHamWOJQ4BiF_MipP5AgDO7RDV2k4pLS4ZEWewKSbJp6JkSLRXRCl13Gai8v8T3cqjj9eq97pW999JHcy5uEc9prXpM8iiNMK1_AVzvsXJASwe28OEXweN7QfuqBVajGotUERQ32OH4TesVdu2OwNCqgLuV_56CHQwg438QH1HXdk_xLSUU7f9Q3pReUSiDmhEwRzQR7n0zHJEha0CFmcu5msqUOzcA9rfoDXzxH8')

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
    return(res) 

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
cred = credentials.Certificate("mypros-283015-firebase-adminsdk-eewnd-fa5c258fcf.json")
firebase_admin.initialize_app(cred)
firebase_storage = pyrebase.initialize_app(config)

db = firestore.client()
storage = firebase_storage.storage()

af = load_model('AF.h5')
brady = load_model('brady.h5')
murmur = load_model('murmur.h5')

app = Flask(__name__)


@app.route("/text", methods=["POST"])
def text():
    name_file = request.values["name"]
    return('hello' + name_file)


@app.route("/heart", methods=["POST"])
def audio():
    data = []
    try:
        all_files = storage.list_files()
        for file in all_files:
            print(file.name)
            file.download_to_filename(file.name)
    except:
        print("An exception occurred")
    h_sound = request.values["heart"]
    user_uid = request.values["uid"]
    url = request.values["url"]
    print(user_uid)
    test1 = extract_data(h_sound)
    spectogram(h_sound, user_uid)
    main_bpm(url,user_uid,h_sound)
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
    returnvalue = [af_return, brady_return, murmur_return]
    return(str(returnvalue))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
