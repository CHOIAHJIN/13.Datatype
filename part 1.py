import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# (산점도,선그래프)
data = pd.read_csv('./data/1.salary.csv')
# 1. 파일을 불러온다
array = data.values
# 2. 불러온 파일을 열별로 나눈다
X = array[:,1]
Y = array[:,2]
# 3. 열을 나눌때 각 열번호에서 하나뺀값으로 넣어서 나눈다
fig, ax = plt.subplots()
#  4. 그림그릴 준비
plt.clf()
#  그림 초기화
plt.scatter(X,Y, label='random',color='gold',marker='*',
            s=30, alpha=0.5)
# 기본으로 주어진 데이터에 대한 산점도 표
X1=X.reshape(-1,1)
#  5. 현제 데이터값이 1대 1이기 때문에 X독립변수에 대해서 행렬로 만들어 주는 함수
# ------------------------------------------------------------------------------------

(X_train,X_text,
 Y_train,Y_test)=train_test_split(X1,Y,test_size=0.2)
# 6. 기본적으로 어떤 데이터를 딥러닝 시키고 어떤데이터를 테스트용으로 뺄지 정하는 단계
model =LinearRegression()
#  선형 분석으로 모델을 만드는 것
model.fit(X_train,Y_train)
#  7. 그 선형 분석의 값을 6번에서 고른 데이터 값으로 만들어라는 명령
y_pred = model.predict(X_text)
#  8. 선형 분석한 값에 6번의 테스트 값을 집어 넣으라는 명령
print(y_pred)

# -------------------------------------------------------------------------
plt.figure(figsize=(10,6))
# 9. 차트의 자체 크기를 정하는 함수
plt.scatter(range(len(Y_test)),Y_test,color='blue',
            marker='o')
# 10.테스트를 한 것의 원본을 점으로 표현해라
plt.plot(range(len(y_pred)),y_pred,color='r'
            ,marker='x')
# 11. 테스트를 한 것의 결과를 선으로 표혀해라
plt.show()

mae=mean_absolute_error(y_pred,Y_test)
print(mae)