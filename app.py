# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:20:10 2020

@author: GOVINDA
"""
import numpy as np
from flask import Flask,request,jsonify,render_template
from sklearn.externals import joblib

app=Flask(__name__)
model, means, stds = joblib.load('new.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features=[float(x) for x in request.form.values()]
    final_features=np.asarray([float_features])
    final_features=(final_features-means)/stds
    prediction=model.predict(final_features)
    if prediction[0]==0:
        output="Non-Diabetic"
    else:
        output="Diabetic"
    #output=round(prediction[0],2)
    
    return render_template('index.html',prediction_text='Patient is {}'.format(output))


if __name__=="__main__":
    app.run(debug=True)
