#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:59:20 2019

@author: zeyuyan
"""

# Import dependencies
import pandas as pd
import numpy as np
from app_backend import Model_Generation
import json
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
from keras import backend as K
from flask import Flask, jsonify, render_template


app = Flask(__name__)


@app.route("/")
def index():
    """Return the homepage."""
    return render_template("index.html")


@app.route("/models")
def models():
    model_list = [
            {"name": "Logistitc Regression", "abbr": "LR"},
            {"name": "K-Nearest Neighbours", "abbr": "KNN"},
            {"name": "Decision Tree", "abbr": "CART"},
            {"name": "Gaussian Naive Bayes", "abbr": "NB"},
            {"name": "Support Vector Machine", "abbr": "SVM"},
            {"name": "Naive Neural Network", "abbr": "naive_NN"},
            {"name": "Reccurent Neural Network (LSTM)", "abbr": "LSTM"}
            ]

    return jsonify(model_list)


@app.route("/dates/<model_abbr>")
def dates(model_abbr):
    with open(f"data/json_files/dates/classifiers/{model_abbr}/dates.json", "r") as load_f:
        dates_list = json.load(load_f)

    return jsonify(dates_list)
    


@app.route("/predict/<model_abbr>/<mm>/<dd>/<yyyy>")
def predict(model_abbr, mm, dd, yyyy):
    date = str(mm) + "/" + str(dd) + "/" + str(yyyy)

    info_dict = {}

    if model_abbr == "LSTM":
        step = 7
    else:
        step = 2

    MG = Model_Generation()
    X_in, Y_to_pred, Y_prev = MG.find_input_output_data_by_date(
            model_abbr,
            date,
            step,
            scaler=MinMaxScaler()
            )

    info_dict["date"] = date
    info_dict["Y_to_pred"] = [Y_to_pred]
    info_dict["Y_prev"] = [Y_prev]

    col_list = MG.generate_col_names()
    col_list.append("DFF")
    X_in_dict = {}
    daily_data_dict = {}
    for i in range(step):
        for j in range(len(col_list)):
            if model_abbr == "LSTM":
                daily_data_dict[col_list[j]] = X_in[0][i][j]
            else:
                daily_data_dict[col_list[j]] = X_in[i * len(col_list) + j]
        X_in_dict["prev_day_" + str(i)] = daily_data_dict


    info_dict["X_in"] = X_in_dict

    if model_abbr == "naive_NN" or model_abbr == "LSTM":
        # Load model
        K.clear_session()
        json_file = open(f'models/classifiers/{model_abbr}/model/model.json', 'r')
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

        # Load weights into loaded model
        loaded_model.load_weights(f'models/classifiers/{model_abbr}/weights/weights.h5')

        # Make predictions
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if model_abbr == "LSTM":
            Y_prob = loaded_model.predict(X_in)
        else:
            X_in_reshaped = X_in.reshape((1, len(X_in)))
            Y_prob = loaded_model.predict(X_in_reshaped)


    else:
        with open(f"models/classifiers/{model_abbr}/model/model.pickle", "rb") as f:
            model = pickle.load(f)

        X_in_reshaped = X_in.reshape((1, len(X_in)))
        Y_prob = model.predict_proba(X_in_reshaped)

    info_dict["Y_prob"] = [float(item) for item in list(Y_prob[0])]

    return jsonify(info_dict)




if __name__ == "__main__":
    app.run(debug=True)
