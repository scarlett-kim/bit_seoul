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
model.fit(x,y, epochs=100, validation_split=0.2)


