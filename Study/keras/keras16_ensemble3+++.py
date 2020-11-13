
#함수형모델 앙상블 + RMSE, R2넣기 
# 데이터가 x 는 2개, y는 1개인 모델 output은 1개 
#1. 데이터 

import numpy as np
#모델1.
x1 = np.array([range(1,101), range(711,811), range(100)])
x2 = np.array([range(4,104), range(761,861), range(100)])

#모델2.
y1 = np.array([range(101,201), range(311, 411), range(100)])


x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.transpose(y1)


#train, test분리 (슬라이싱 대신에 train_test_split)
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, shuffle=True, train_size =0.7)
x2_train, x2_test = train_test_split(x2,shuffle=True, train_size =0.7)

print("x1_test", x1_test.shape) 
print("x2_test", x2_test.shape) 
print("y1_test", y1_test.shape)

#함수형 모델 만들기 
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input

#모델1. 
input1 = Input(shape =(3,))
dense1 = Dense(10, activation='relu', name='king1')(input1) #모델 이름이 중복될 때 , 혹은 dense이름 변경
dense2 = Dense(7, activation='relu', name='king2')(dense1)
dense3 = Dense(5, activation='relu', name='king3')(dense2)
output1 = Dense(3, activation ='linear', name='king4')(dense3)

#모델2.
input2 = Input(shape =(3,))
dense1_1 = Dense(15, activation='relu')(input2)
dense2_1 = Dense(11, activation='relu')(dense1_1)
output2 = Dense(3, activation ='linear')(dense2_1)


#모델끼리는 별 영향을 끼치지 않는다 
##########모델 병합, concantenate 여러가지 방법으로import 
from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import Concatenate, concatenate
# from keras.layers import Concatentate, concatenate

#########모델 2개 병합하기 
#소문자 concatenate
#merge1 = concatenate([output1, output2])#리스트로 묶기

#방법 1. Concatenate대문자 사용법
#merge1 = Concatenate()([output1, output2]) #대문자 Concatenate 면 class다 
#방법 2. 
merge1 = Concatenate(axis=1)([output1, output2]) #axis축 // 여기서의 output은 y축의 데이터 개수가 아니고 모델 1의output, 모델2의output
middle1 = Dense(30)(merge1) #함수형으로 모델 병합하기 
middle1 = Dense(7)(middle1)
middle1 = Dense(11)(middle1)

'''
merge1 = concatenate([output1, output2])#리스트로 묶기
#앞에 변수명 같이 명시 해도 가능. 
middle1 = Dense(30)(merge1) #함수형으로 모델 병합하기 
middle2 = Dense(7)(middle1)
middle3 = Dense(11)(middle2) #이렇게 middle1도 가능? 
'''
########### output 만들기 (분기) y갯수에 따라서 
output3 = Dense(30)(middle1)#middle1에서 받아온다 
output3_1 = Dense(7)(output3)
output3_2 = Dense(3)(output3_1)#outputs에 마지막 아웃풋 넣기 

# output4 = Dense(15)(middle1)
# output4_1 = Dense(14)(output4)
# output4_2 = Dense(11)(output4_1)
# output4_3 = Dense(3)(output4_2)#마지막 아웃풋 넣기 

#모델 정의하기 
model = Model (inputs =[input1, input2], outputs =output3_2)
                        #모델 1,2넣기               #ㄴ여기아웃풋은 input할 때의 output이 아니라 output할 때 마지막 output변수명 
model.summary()

#concatenate는 연산하지 않고 병합만 해준다 

# 컴파일, 훈련 
model.compile(loss= 'mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y1_train, epochs=100, batch_size=8, validation_split=0.25, verbose=1)

# 평가 
result = model.evaluate([x1_test, x2_test], y1_test, batch_size=8)
# (loss, 메트릭스의 mse)가 나온다 

print('result:' , result) 
#result: [1.739203691482544, 0.4144916236400604, 1.3247121572494507, 0.47640863060951233, 0.80832839012146]

#예측 = 훈련시킨 값이 나온다 (accuracy 의 정확도를 위해서 예측이 필요하다) x값을 넣어서 예측 y값을 나오게
y_predict = model.predict([x1_test, x2_test]) #y_predict라는 변수에 위에서 validation_split으로 쪼갠 test데이터에서 y값을 예측하자 
print('결과물: \n :' , y_predict) 


#RMSE
#사이킷런 지표
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y_predict) : #y1_test validation_split값의 y예측 테스트 데이터를 가져오고, 예측 데이터를 넣을 변수명
    return np.sqrt(mean_squared_error(y1_test, y_predict))
print('RMSE:' ,RMSE(y1_test, y_predict))

#R2 
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y_predict)
print('R2:', r2)