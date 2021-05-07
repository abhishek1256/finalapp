import flask
from flask import Flask, jsonify, request,render_template, request, redirect, url_for
import json
import tensorflow as tf
import librosa
import numpy as np
from scipy.io.wavfile import write
import os
# -*- coding: utf-8 -*-
'''
Script for creating and loading contents to the server
'''
import flask
from flask import Flask, jsonify, request
import json
import tensorflow as tf
import librosa
import numpy as np
from scipy.io.wavfile import write
import tensorflow.keras.backend as K
import soundfile

def load_model():
    K.clear_session()
    model = tf.keras.models.load_model(r"gbl_model.h5", compile=False)
    return model

def inputProcess(filepath, A=2000, L=110):
    arr, _ = librosa.load(filepath, sr=22000, duration=10)
    #arr = open(filepath, "r")
    print("array = ",arr)

    arr_pad = np.pad(arr, (0, A*L - len(arr)), 'constant', constant_values=(0,0))
    arr_reshaped = arr_pad.reshape(1, A, L, 1)
    #print("arr_reshaped= ",arr_reshaped)
    return arr_reshaped

def wavCreator(path, arr):
    arr = np.array(arr).T
    #librosa.output.write_wav(path, arr, sr=22000)
    soundfile.write(path, arr, 22000)
    #write(path, 22000, arr)
    

app = Flask(__name__)
model=load_model()
@app.route('/')
def index():
    return render_template('recent.html')

@app.route('/', methods=['POST'])
def upload_file():
    
    if "file" not in request.files:
        return redirect(request.url)

    
    file = request.files["file"]
            
    if file.filename == "":
        return redirect(request.url)
    if file:
       #file.save(file.filename)
        arr_reshaped = inputProcess(file)
    print("file is:",file)
    
    denoised_arr = model.predict([arr_reshaped, np.zeros((1, 2000*110))])
    path=r"static"
    path=path+'\TDB_Prediction.flac'
        
    wavCreator(path, denoised_arr)
    response = json.dumps({1:2})

    return redirect(url_for('index'))

   
if __name__ == '__main__':
   app.run(debug = True)
