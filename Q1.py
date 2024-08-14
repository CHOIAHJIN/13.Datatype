import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix

data = pd.read_csv('./data/1.salary.csv')

array = data.values
X = array[:,0]
Y = array[:,1]
X = X.reshape(-1,1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

model =LinearRegression()
model.fit(X_train,Y_train)
model.coef_
model.intercept_

# 모델 예측
y_pred = model.predict(X_test)
error = mean_absolute_error(y_pred, Y_test)
print(error)

plt.clf()
plt.scatter(X_test, Y_test,color = 'blue', label = 'Actual values')
plt.plot(range(len(y_pred)), y_pred, color = 'red', label = 'Predicred values', marker = 'o')
plt.legend()
plt.xlabel("Experience Years")
plt.ylabel("Salary  ($)")
plt.show()




