# 데이터 shape
#multi layer 퍼섹트롬
import numpy as np

x = np.array([range(1,101), range(311,411), range(100)])
y = np.array([range(101,201), range(711, 811), range(100)])

print(x) #100개의 데이터 3개 
print("transpose하기전" , x.shape)#(3,10) #(3,) 스칼라 30개 = 객체 inputing=1 / vector=1차원/ 매트릭스(행렬)=2차원 / 텐서 =3차원
#나는 shape가 (100,3) 이 나오길 바란다 

#(100,3) 형태로 x,y를 변환시켜라 

#행무시, 열우선
'''
위와 같이 쓸 경우  (3, 100)
전치를 할 경우 필요한 것 transpose , reshape 
''' 

x= np.transpose(x)
y= np.transpose(y)
print("transpose 하고 난 후" ,x.shape) #(100,3)

#y1, y2, y3 = w1*1 + w2*2 + w3*3 +b
#모델 구성하기 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim =3))
model.add(Dense(5))
model.add(Dense(3)) #출력 컬럼 3개 

#나머지 완성 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size =0.7, shuffle=False)
#x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size =0.3, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size =0.2, shuffle=False) 

# 순서 3. 컴파일 , 화면 
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train, y_train, epochs=100 , validation_split=0.2)

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

