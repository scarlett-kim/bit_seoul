#R2 맥스값 1 ,회귀지표 , 회귀에서 accuracy로 사용한다

import numpy as np

#RMSE지표 Root mean square error :평균 제곱근 오차
#회귀 지표: 제공되는 것(mse, mae), RMSE(사용자정의로 만든 것) 

#순서 1. x와 y 의 데이터 구성 
 
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 훈련시킬 변수를 train이라고 칭한다  
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15]) # 평가할 데이터는 test
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18]) # 예측할 데이터는 predict , 기본 x값을 넣으면y 값이 나온다 


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

# 순서 2. 모델 구성 (모델 하이퍼 파라미터 튜닝:소스를 수정하지 않고 다시 훈련할 때 결과값이 달라지는 경우: 맞다.)
model = Sequential() #weight가중치값을 지정해서 매번 같은 값을 나오게 할 수 있다 
model.add(Dense(3, input_dim =1)) 
model.add(Dense(50))
model.add(Dense(30))  
model.add(Dense(7))
model.add(Dense(1)) # 노드 개수가 많다고 무조건 연산이 좋아지진 않는다.



# 순서 3. 컴파일 , 화면 (수정가능하다 = 하이퍼 파라미터 튜닝)
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 


model.fit(x_train, y_train, epochs=10000) 

# 순서 4. 평가, 예측
#loss, acc = model.evaluate(x, y) 
loss = model.evaluate(x_test, y_test)  


print("loss: ", loss) 
# print("acc:" , acc)


#예측 = 훈련시킨 값이 나온다 (accuracy 의 정확도를 위해서 예측이 필요하다)
y_predict = model.predict(x_test) #x_test를 넣는 이유는 지표를 찾기 위해서 원래는 y_predict = model.predict(x_pred)

print('결과물: \n :' , y_predict) 

#RMSE는 파라미터말고 아래 추가해야한다
#사이킷런 지표
from sklearn.metrics import mean_squared_error
# 사이킷런에 metrics mse 제공
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
print('RMSE:' , RMSE(y_test, y_predict)) #RMSE: 1.4145276045028044e-06

#R2 , accuracy대용으로 쓸 수 있다 RMSE랑 같이 써야 좋다 서로 보완하는 보완지표 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2:', r2) #R2: 0.9999999999995453


