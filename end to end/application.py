import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pickle

# Creating an instance of the Flask class
application = Flask(__name__)
app = application

# Load Ridge Regressor model and Standard Scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Get input data from form
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get("Rain"))
            FFMC = float(request.form.get("FFMC"))
            DMC = float(request.form.get("DMC"))
            ISI = float(request.form.get("ISI"))
            Classes = float(request.form.get("Classes"))
            Region = float(request.form.get("Region"))

            # Transform input data
            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

            # Make prediction
            result = ridge_model.predict(new_data_scaled)

            # Pass result to template
            return render_template("home.html", results=result[0])

        except Exception as e:
            return f"<h2>Error: {str(e)}</h2>"

    # If GET request, show the form without result
    return render_template("home.html", results=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

    