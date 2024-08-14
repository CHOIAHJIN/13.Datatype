# Q2
# sepal-length : 꽃받침의 길이, sepal-width : 꽃받침의 너비
# petal-length : 꽃잎의 길이, petal-width : 꽃잎의 너비, class : 꽃의 종류
# (1) 주어진 데이터 셋에 대한 정보 파악(데이터 요약(describe()) 및 시각화)
# (2) 꽃받침 및 꽃잎의 정보를 바탕으로 꽃의 종류를 예측하는 의사결정나무 모델 개발
# (3) 개발한 모델의 성능 평가 : KFold 교차검증 방법을 활용(정확도)
# (4) 개발한 모델의 성능 평가 : 오차행렬, 정밀도, 재현율 등

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