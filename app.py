import csv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
import pickle
import flask
import json


grade10_courses = ['AP Biology', 'AP Calculus AB', 'AP Computer Science Principles', 'AP Microeconomics', 'AP Physics 1', 'AP Physics 2', 'AS English Literature', 'C-Humanities', 'Chinese II', 'Chinese as a Second Language II', 'English II', 'Fine Art I', 'IGCSE Additional Math', 'IGCSE Biology (Accelerated)', 'IGCSE Chemistry (Accelerated)', 'IGCSE Economics (Year 2)', 'IGCSE English Literature (Year 2)', 'IGCSE History (Year 2)', 'IGCSE Math (Year 2)', 'IGCSE Physics (Accelerated)', 'IT (Year 1)', 'World History II']
grade9_courses = ['AP 2D Art and Design', 'AP Computer Science Principles', 'AP Music Theory', 'AP Physics 1', 'Art Design', 'C-Humanities', 'CSL', 'Chinese Culture', 'Drama I', 'English I', 'IGCSE Biology (Accelerated)', 'IGCSE Combined Science (Year 1)', 'IGCSE Economics (Year 1)', 'IGCSE English Literature (Year 1)', 'IGCSE History (Year 1)', 'IGCSE Math (Accelerated)', 'IGCSE Math (Year 1)', 'IT (Year 1)', 'IT (Year 2)', 'Music', 'PE', 'Spanish I', 'Spanish II', 'World History I']
mean = {
    "AP 2D Art and Design": 95.0,
    "AP Computer Science Principles": 95.85714285714286,
    "AP Music Theory": 93.0,
    "AP Physics 1": 90.4,
    "Art Design": 95.26666666666667,
    "C-Humanities": 87.48611111111111,
    "CSL": 92.66666666666667,
    "Chinese Culture": 95.29166666666667,
    "Drama I": 88.54545454545455,
    "English I": 77.72,
    "IGCSE Biology (Accelerated)": 91.18181818181819,
    "IGCSE Combined Science (Year 1)": 88.575,
    "IGCSE Economics (Year 1)": 84.61702127659575,
    "IGCSE English Literature (Year 1)": 89.09433962264151,
    "IGCSE History (Year 1)": 89.42857142857143,
    "IGCSE Math (Accelerated)": 92.57142857142857,
    "IGCSE Math (Year 1)": 86.16666666666667,
    "IT (Year 1)": 89.66666666666667,
    "IT (Year 2)": 92.91666666666667,
    "Music": 94.46428571428571,
    "PE": 93.85897435897436,
    "Spanish I": 97.0,
    "Spanish II": 96.6923076923077,
    "World History I": 78.625
}

# Load the training data used to train the model
app = flask.Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])
def predict( ):
    
    

    print('got request')

    # Load the models   
    models = pickle.load(open('model.sav','rb'))
    
    # Get the input grades from the request data
    score_list = flask.request.json
    
    print(score_list)

    # Convert the input grades to a 2D array with a single row
    score_list = [float(x) if x != '' else 0.0 for x in score_list]
    X_test = np.array(score_list).astype(float).reshape(1, -1)
    
    # Fill in the null Score with the data given above
    for i in range(len(X_test[0])):
        if X_test[0][i] == 0.0:
            X_test[0][i] = mean[grade9_courses[i]]

    scores = {}
    for subject in grade10_courses:
        model = models[subject]
        score = model.predict(X_test)
        scores[subject] = score.tolist()[0]
    print(scores)
    return scores

