import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Define grade to grade mapping for the courses
grade9_courses = ['AP 2D Art and Design', 'AP Computer Science Principles', 'AP Music Theory', 'AP Physics 1', 'Art Design', 'C-Humanities', 'CSL', 'Chinese Culture', 'Drama I', 'English I', 'IGCSE Biology (Accelerated)', 'IGCSE Combined Science (Year 1)', 'IGCSE Economics (Year 1)', 'IGCSE English Literature (Year 1)', 'IGCSE History (Year 1)', 'IGCSE Math (Accelerated)', 'IGCSE Math (Year 1)', 'IT (Year 1)', 'IT (Year 2)', 'Music', 'PE', 'Spanish I', 'Spanish II', 'World History I']
grade10_courses = ['A Level Further Math', 'AP Biology', 'AP Calculus AB', 'AP Calculus BC', 'AP Chemistry', 'AP Computer Science Principles', 'AP Microeconomics', 'AP Physics 1', 'AP Physics 2', 'AS English Literature', 'C-Humanities', 'Chinese II', 'Chinese as a Second Language II', 'English II', 'Fine Art I', 'IGCSE Additional Math', 'IGCSE Biology (Accelerated)', 'IGCSE Chemistry (Accelerated)', 'IGCSE Economics (Year 2)', 'IGCSE English Literature (Year 2)', 'IGCSE History (Year 2)', 'IGCSE Math (Year 2)', 'IGCSE Physics (Accelerated)', 'IT (Year 1)', 'Music I', 'Music II', 'Physical Education', 'Spanish I', 'World History II']
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

print(X)
print(Y)
# Fill in the null Score
imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)

# Fit the model in with XGboost and predict the grades


# Create a dictionary to store the models for each subject
models = {}

for subject in grade10_courses:
    # Fit the model
    model = xgb.XGBRegressor()
    model.fit(X, Y.loc[:, grade10_courses.index(subj)])
    # Store the model
    models[subject] = model


# Load the test data
test = pd.read_csv('student_g9.csv')
test = test.replace('N', np.nan)
test = test.replace(grade_dict)
test = imputer.fit_transform(X)

# Predict the grades
for subject in grade10_courses:
    test[:, grade10_courses.index(subject)] = models[subject].predict(test)

# Save the result
test = pd.DataFrame(test, columns=grade10_courses)
test.to_csv('student_g10.csv', index=False)