import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
# Importing the dataset
dataset = pd.read_csv('Pos_Sal.csv')
pos = dataset.iloc[:, 1:2].values
lab = dataset.iloc[:, 2].values.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(pos, lab, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
x_train = scx.fit_transform(x_train)
x_test = scx.transform(x_test)
scy = StandardScaler()
y_train = scy.fit_transform(y_train)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
reg.fit(pos, lab)

# Predicting a new result
y_pred = reg.predict(6.5)
print("Prediction of random tree regression: ",y_pred)
train_val = reg.score(x_train,y_train)
test_val = reg.score(x_test,y_test)
print("Accuracy on train set: ",train_val)
print("Accuracy on test set: ",test_val)
# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(pos), max(pos), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(pos, lab, color = 'red',label = "Real value")
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
plt.scatter(6.5,y_pred,color="orange",label="Predicted value")
plt.title('Random Forest Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()
plt.show()