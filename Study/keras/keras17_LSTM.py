# DNN : Deep Neural Network => 은닉 레이어가 2개 이상인 신경망
# RNN : Recurrent Neural Network => 순환신경망. 순차적인 데이터 처리 => Time Series 시계열
# LSTM : RNN에서 가장 성능 좋은 기법

#1. 데이터
import numpy as np 

x= np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) #(4,3)
y= np.array([4,5,6,7]) #(4,)

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape(x.shape[0], x.shape[1],1)
# x= x.reshape(x.shape[0], x.shape[1], 1)  #x의 자료를 하나씩 잘라 연산하도록([[[1],[2],[3]], [[2],[3],[4]],[[3],[4],[5]],[[4],[5],[6]]]) / (4,3)=>(4,3,1)로 변환 
                                        #LSTM 행x열x몇개씩 잘라 작업하는지(자르는 크기) 
#x=x.reshape(4,3,1)
print("x.shape : ", x.shape)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model=Sequential()
model.add(LSTM(20, activation='relu', input_shape=(3,1)))   
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(1))

model.summary()
#LSTM params 
# = 4 * ((size_of_input + 1) * size_of_output + size_of_output^2)  = 4 * ((1+1) * 10 + 100) = 480
# = 4 * ((자른 입력 갯수 + biases + 노드갯수) * 노드갯수 = 4 * (1+1+10) * 10 = 480
# 시계열은 Dense에 비해 연산이 많고 그만큼 연산을 잘 하지만 속도가 느림

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.3)

x_input=np.array([5,6,7]) #(3,) ->(1,3,1) 입력에 형변환이 있었으므로 예측값을 내는 자료도 형변환을 해줘야 함. [5,6,7]을 [[5], [6], [7]] 형태로....
x_input= x_input.reshape(1,3,1)


# loss=model.evaluate(x_test, y_test, batch_size=1)

y_predict=model.predict(x_input)

# print("loss : ", loss)
print("y_predict : ", y_predict)


#LSTM은 자료를 잘라서 잘라낸 자료에서 다음 자료로 넘어갈 때마다 연산을 한다. 
#[2,3,4]가 있고 이를 하나씩 잘라준다고 하면 2에서 3으로 넘어갈 때 +1, 3에서 4로 갈때 +1, 그렇게 y=[5]가 나오도록 함. 때문에 연산할 자료의 수를 잘라서 형변환을 해 줘야됨.