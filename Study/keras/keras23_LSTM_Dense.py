# 여태 LSTM으로 구성한 모델을 Dense로 만들어서 성능 비교하기

import numpy as np     

x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],
            [8,9,10],[9,10,11], [10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_input=np.array([50,60,70])
x_input=x_input.reshape(1,3)

# x=x.reshape(13,3,1)
# x_input=x_input.reshape(1,3,1)

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(200, activation='relu', input_dim=3))
model.add(Dense(150, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

model.summary()


model.compile(loss='mse', optimizer='adam')


# from tensorflow.keras.callbacks import EarlyStopping

# early_stopping=EarlyStopping(monitor='loss', patience=50, mode='min')

model.fit(x, y, epochs=100, batch_size=1)

y_predict=model.predict(x_input)

print("y_predict : ", y_predict)
