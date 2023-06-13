import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
#import random forest
from sklearn.ensemble import RandomForestRegressor
import pickle


# Define grade to grade mapping for the courses
grade9_courses = ['AP 2D Art and Design', 'AP Computer Science Principles', 'AP Music Theory', 'AP Physics 1', 'Art Design', 'C-Humanities', 'CSL', 'Chinese Culture', 'Drama I', 'English I', 'IGCSE Biology (Accelerated)', 'IGCSE Combined Science (Year 1)', 'IGCSE Economics (Year 1)', 'IGCSE English Literature (Year 1)', 'IGCSE History (Year 1)', 'IGCSE Math (Accelerated)', 'IGCSE Math (Year 1)', 'IT (Year 1)', 'IT (Year 2)', 'Music', 'PE', 'Spanish I', 'Spanish II', 'World History I']
grade10_courses = ['AP Biology', 'AP Calculus AB', 'AP Chemistry', 'AP Computer Science Principles', 'AP Microeconomics', 'AP Physics 1', 'AP Physics 2', 'AS English Literature', 'C-Humanities', 'Chinese II', 'Chinese as a Second Language II', 'English II', 'Fine Art I', 'IGCSE Additional Math', 'IGCSE Biology (Accelerated)', 'IGCSE Chemistry (Accelerated)', 'IGCSE Economics (Year 2)', 'IGCSE English Literature (Year 2)', 'IGCSE History (Year 2)', 'IGCSE Math (Year 2)', 'IGCSE Physics (Accelerated)', 'IT (Year 1)', 'World History II']
grade_dict = {'A+': 97, 'A': 93, 'A-': 90, 'B+': 87, 'B': 83, 'B-': 80,'C+': 77, 'C': 73, 'C-': 70, 'D+': 67, 'D': 63, 'D-': 60, 'F': 0}

# Load the data
X = pd.read_csv('X_g9.csv')
Y = pd.read_csv('X_g10.csv')

# replace N to 0
X = X.replace('N', np.nan)
Y = Y.replace('N', np.nan)
# replace the grades to numbers
X = X.replace(grade_dict)
Y = Y.replace(grade_dict)

# Fill in the null Score
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)


# Define a dictionary to store the models for each grade 10 subject
models = {}


# Train a randomforest model for each grade 10 subject
for subject in grade10_courses:
    index = grade10_courses.index(subject)
    Y_scores = Y[:, index]
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
    # Fit the model
    model.fit(X, Y_scores)
    # Store the model
    models[subject] = model
# Save the models to disk
filename = 'model.sav'
pickle.dump(models, open(filename, 'wb'))

# Load the models from disk
# filename = 'model.sav'
# models = pickle.load(open(filename, 'rb'))


# Load one student for Testing
X_test = pd.read_csv('student_g9.csv')
X_test = X_test.replace('N', np.nan)
X_test = X_test.replace(grade_dict)
imputer = SimpleImputer(strategy='mean')
imputer.fit(X)

# Use imputer to fill in missing values in X_test with most frequent values from X
X_test = imputer.transform(X_test)
print(X_test)



scores = {}
for subject in grade10_courses:
    index = grade10_courses.index(subject)
    model = models[subject]
    score = model.predict(X_test)
    scores[subject] = score[0]
    # jump the courses that the student had taken in grade 9 or the student had already take igcse math accelerated in grade 9, which means the student will not take igcse math in grade 10 IT Class is excluded
    

print(scores)
# Write the scores to a CSV file using Pandas
df = pd.DataFrame([scores], index=['Scores'])
df.to_csv('scores.csv', index=True, header=True)

