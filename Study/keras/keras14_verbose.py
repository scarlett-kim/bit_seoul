# 데이터 shape
import numpy as np

x = np.array([range(1,101), range(711,811), range(100)])
y = np.array(range(101,201))

print(x) 
print("transpose하기전" , x.shape) #transpose하기전 (3, 100)
print("t" , y.shape) #(100, )

x= np.transpose(x)
y= np.transpose(y)
print("transpose 하고 난 후" ,x.shape) #transpose 하고 난 후 (100, 3)

#사이킷런사용하여 트레인스플릿으로 슬라이싱저절로 되게 한다 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle =False, train_size=0.7)

#모델 구성 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(10, input_dim =(3, )))
model.add(Dense(10, input_shape =(3, )))
#(100,10, 3): input_shape(10,3) 행무시 ?
model.add(Dense(5))
model.add(Dense(3))

#컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
#model.fit(x_train, y_train, epochs=100, validation_split=0.25, verbose=2) #verbose라는 파라미터
model.fit(x_train, y_train, epochs=100, validation_split=0.25, verbose=0) #verbose중간에 실행과정이 안 나온다 loss: 3858.9939 - mae: 53.5642
model.fit(x_train, y_train, epochs=100, validation_split=0.25, verbose=1) 
# loss: 506.1399 - mae: 19.6918 - val_loss: 1417.1885 - val_mae: 32.8744 Epoch 96/100
model.fit(x_train, y_train, epochs=100, validation_split=0.25, verbose=3) 


#평가, 예측
loss,mse =model.evaluate(x_test, y_test)
print(loss)
print(mse)

y_predict = model.predict(x_test)
print(y_predict)

#RMSE

#R2




