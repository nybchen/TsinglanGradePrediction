import pandas as pd
import numpy as np
import flask
import csv
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from flask_cors import CORS

grade10_courses = ['AP Biology', 'AP Calculus AB', 'AP Chemistry', 'AP Computer Science Principles', 'AP Microeconomics', 'AP Physics 1', 'AP Physics 2', 'AS English Literature', 'C-Humanities', 'Chinese II', 'Chinese as a Second Language II', 'English II', 'Fine Art I', 'IGCSE Additional Math', 'IGCSE Biology (Accelerated)', 'IGCSE Chemistry (Accelerated)', 'IGCSE Economics (Year 2)', 'IGCSE English Literature (Year 2)', 'IGCSE History (Year 2)', 'IGCSE Math (Year 2)', 'IGCSE Physics (Accelerated)', 'IT (Year 1)', 'World History II']
grade9_courses = ['AP 2D Art and Design', 'AP Computer Science Principles', 'AP Music Theory', 'AP Physics 1', 'Art Design', 'C-Humanities', 'CSL', 'Chinese Culture', 'Drama I', 'English I', 'IGCSE Biology (Accelerated)', 'IGCSE Combined Science (Year 1)', 'IGCSE Economics (Year 1)', 'IGCSE English Literature (Year 1)', 'IGCSE History (Year 1)', 'IGCSE Math (Accelerated)', 'IGCSE Math (Year 1)', 'IT (Year 1)', 'IT (Year 2)', 'Music', 'PE', 'Spanish I', 'Spanish II', 'World History I']

# Load the training data used to train the model
X = pd.read_csv('X_G9.csv')

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
    X_test = np.reshape(score_list, (1, len(score_list)))
    score_list = [float(x) if x != '' else 0.0 for x in score_list]
    X_test = np.array(score_list).astype(float).reshape(1, -1)
    print(X_test)

    # Add a header of all thegrade 9 courses to the input grades
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    print(X_test)

    # Use imputer to fill in missing values in X_test with most frequent values from X
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(X)
    X_test = imputer.transform(X_test)

    # Make prediction
    scores = {}
    for subject in grade10_courses:
        model = models[subject]
        score = model.predict(X_test)
        scores[subject] = score[0]
    print(scores)
    return scores