from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipline import CustomData, PredictPipeline

application=Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictions', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            ct_depth=float(request.form.get('ct_depth')),
            ct_pressure=float(request.form.get('ct_pressure')),
            n2_rate=float(request.form.get('n2_rate'))
            
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)