
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# 2.모델 
# model = Sequential()
# model.add (LSTM (100, input_shape = (4,1)))
# model.add (Dense(50, name ='queen1'))
# model.add (Dense(10, name ='queen2'))
# # model.add (Dense(1, name ='queen3')) 29번에 추가되는 모델 
 
# model.summary()

model=Sequential()
model.add(LSTM(20, input_shape=(3,1)))    #30번의 input_shape와 상관없이
model.add(Dense(30, name ='joker1'))
model.add(Dense(100, name ='joker2'))

model.summary()
'''
dense (Dense)                (None, 50)                5050
_________________________________________________________________
dense_1 (Dense)              (None, 10)                510
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 46,371
Trainable params: 46,371
Non-trainable params: 0
_________________________________________________________________
'''
# 저장 하기 
model.save("save.1.h5") #확장자 h5파일에 저장하겠다 => Study라는 폴더 안에 넣어야 한다 
#ㄴ이렇게 저장하면 별로니까 

model.save("./save/keras.30.h5") # .은 Study폴더 안에 save폴더안에 keras.28폴더를 
# model.save(".\save\keras28_2.h5") #\n은 개행이라 오류가 난다 
# model.save(".//save//keras28_3.h5")
# model.save(".\\save\\keras28_4.h5")
