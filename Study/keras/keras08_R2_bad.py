# 실습 개판으로 만들어라 
# R2를 음수가 아닌 0.5 아래로 줄이기
# 레이어는 인풋과 아웃풋 포함 7개 이상(히든이 5개 이상)
# 히든레이어 노드는 레이어당 각각 최소 10개 이상
# batch_size =1
# epochs = 100이상
# 데이터 조작 금지 

import numpy as np

#순서 1. x와 y 의 데이터 구성 
 
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15]) 
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18]) 

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

# 순서 2. 모델 구성 : 이렇게 모델을 구성하면 나쁘다
model = Sequential() 
model.add(Dense(3, input_dim =1)) 
model.add(Dense(5000))
model.add(Dense(10))  
model.add(Dense(10000))
model.add(Dense(10))
model.add(Dense(500000))
model.add(Dense(10)) 
model.add(Dense(10000)) 
model.add(Dense(1)) 
model.add(Dense(5000))
model.add(Dense(10))  
model.add(Dense(10000))
model.add(Dense(10))
model.add(Dense(500000))
model.add(Dense(10)) 
model.add(Dense(10000)) 
model.add(Dense(1)) 


# 순서 3. 컴파일 , 화면 (수정가능하다 = 하이퍼 파라미터 튜닝)
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 


model.fit(x_train, y_train, epochs=100, batch_size=1) 

# 순서 4. 평가, 예측

loss = model.evaluate(x_test, y_test, batch_size=1)  


print("loss: ", loss) 


#예측 = 훈련시킨 값이 나온다 (accuracy 의 정확도를 위해서 예측이 필요하다)
y_predict = model.predict(x_test) 
print('결과물: \n :' , y_predict) 

#RMSE는 파라미터말고 아래 추가해야한다
#사이킷런 지표
from sklearn.metrics import mean_squared_error
# 사이킷런에 metrics mse 제공
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE:' , RMSE(y_test, y_predict)) 

#R2 , accuracy대용으로 쓸 수 있다 RMSE랑 같이 써야 좋다 서로 보완하는 보완지표 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2:', r2) #R2: 0.9999999999995453


