from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)

ridge_model = pickle.load(open('E:\Algerian Forest Fires Project\Models\\ridge.pkl','rb'))
standard_scaler = pickle.load(open('E:\Algerian Forest Fires Project\Models\scaler.pkl','rb'))

#route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature = int(request.form.get('Temperature'))
        RH = int(request.form.get('RH'))
        Ws =int(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = int(request.form.get('Classes'))
        Region = int(request.form.get('Region'))

        new_data_scale = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scale)

        return render_template('home.html',result=result[0])
    
    else:
        return render_template('home.html')

if __name__=='__main__':
    app.run(host="0.0.0.0")