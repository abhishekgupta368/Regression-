#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
#Importing the dataset
dataset = pd.read_csv('Pos_Sal.csv')
pos = dataset.iloc[:, 1:2].values
label = dataset.iloc[:, 2].values.reshape(-1,1)

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(pos, label, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

#Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(pos, label)
train_res = regressor.score(x_train,y_train)
test_res = regressor.score(x_test,y_test)

print("Accuracy of train set: ",train_res)
print("Accuracy of test set: ",test_res)
#Predicting a new result
y_pred = regressor.predict(y_test)
sco = r2_score(y_test,y_pred)
print("R2 score: ",sco)
#Visualising the Decision Tree Regression results (higher resolution)
x_grid = np.arange(min(pos), max(pos), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(pos, label, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Salaries Graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()