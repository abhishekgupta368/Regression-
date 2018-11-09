# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('startups.csv')
info = dataset.iloc[:, :-1].values
lab = dataset.iloc[:, 4].values.reshape(-1,1)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lab_enc = LabelEncoder()
info[:, 3] = lab_enc.fit_transform(info[:, 3])
one_enc = OneHotEncoder(categorical_features = [3])
info = one_enc.fit_transform(info).toarray()

# Avoiding the Dummy Variable Trap
info = info[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(info, lab, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
X_train = scx.fit_transform(X_train)
X_test = scx.transform(X_test)
scy = StandardScaler()
y_train = scy.fit_transform(y_train)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
train_res = regressor.score(X_train,y_train)
test_res = regressor.score(X_test,y_test)
sco = r2_score(y_test,y_pred)
print("Accuracy of train set: ",train_res)
print("Accuracy of test set: ",test_res)
print("R2 score: ",sco)
