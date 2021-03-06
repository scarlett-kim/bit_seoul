#1. 데이터 valication_data 으로 검증데이터 +슬라이싱
import numpy as np
x = np.array(range(1,101)) #weight =1, 바이어스=100 , 1~100까지
y = np.array(range(101,201)) # 101~200까지

#데이터 슬라이싱
x_train = x[:60] # : 처음부터 60까지 
y_train = y[:60]
x_val = x[61:80]
y_val = y[61:80] #20개
x_test = x[20:] #20개   
y_test = y[20:]

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

model = Sequential() 
model.add(Dense(3, input_dim =1)) 
model.add(Dense(50))
model.add(Dense(30))  
model.add(Dense(7))
model.add(Dense(1)) 

# 순서 3. 컴파일 , 화면 
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 

model.fit(x_train, y_train, epochs=100 , validation_data=(x_val, y_val))

# 순서 4. 평가, 예측
#loss, acc = model.evaluate(x, y) 
loss = model.evaluate(x_test, y_test)  


print("loss: ", loss) 

#예측 = 훈련시킨 값이 나온다 (accuracy 의 정확도를 위해서 예측이 필요하다)
y_predict = model.predict(x_test) #x_test를 넣는 이유는 지표를 찾기 위해서 원래는 y_predict = model.predict(x_pred)

print('결과물: \n :' , y_predict) 

#RMSE는 파라미터말고 아래 추가해야한다
#사이킷런 지표
from sklearn.metrics import mean_squared_error
# 사이킷런에 metrics mse 제공
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE:' , RMSE(y_test, y_predict)) #0.20419436496650925

#R2 , accuracy대용으로 쓸 수 있다 RMSE랑 같이 써야 좋다 서로 보완하는 보완지표 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2:', r2) #R2: 0.979152330657962