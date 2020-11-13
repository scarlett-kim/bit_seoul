# 데이터 shape
# 1개가지고 3개의 컬럼을 찾기 
import numpy as np

x = np.array(range(1,101))
y = np.array([range(101,201), range(711, 811), range(100)])

print(x) 
print("transpose하기전" , x.shape) #transpose하기전 (100,)
print("transpose하기전" , y.shape) #transpose하기전 (3, 100)


x= np.transpose(x)
y= np.transpose(y)
print("transpose 하고 난 후" ,x.shape) #transpose 하고 난 후 (100,)
print("transpose 하고 난 후" ,y.shape) #transpose 하고 난 후 (100, 3)


#모델 구성하기 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim= 1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))

#사이킷 런 사용하여 train_test_split을 사용하여 슬라이싱 할 필요 없이, 짧은 코드로 70%, 20% 이런식으로 나눌 수 있다 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.7, shuffle=False) #train이 70%
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.2, shuffle=False)

#컴파일 화면
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=100 ,validation_split=0.2)

#평가 예측
loss = model.evaluate(x_test, y_test)
print("loss값은?",loss)





