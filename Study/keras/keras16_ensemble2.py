
#y 아웃풋이 3개인 모델 
#1. 데이터 

import numpy as np
#모델1.
x1 = np.array([range(1,101), range(711,811), range(100)])
x2 = np.array([range(4,104), range(761,861), range(100)])

#모델2.
y1 = np.array([range(101,201), range(311, 411), range(100)])
y2 = np.array([range(501,601), range(431,531), range(100,200)])
y3 = np.array([range(501,601), range(431,531), range(100,200)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)

x2 = np.transpose(x2)
y2 = np.transpose(y2)

y3 = np.transpose(y3)

#train, test분리 (슬라이싱 대신에 train_test_split)
from sklearn.model_selection import train_test_split 
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, shuffle=True, train_size =0.7)
x2_train, x2_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x2, y2, y3, shuffle=True, train_size =0.7)
#++++ y3 연결하기 + 위처럼 3개까지는 가능하지만 그 이상은 추가로 train_test_split별로 분리 
# y3_train, x3_test = train_test_split(y3, shuffle=True, train_size =0.7)


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
merge1 = Concatenate(axis=1)([output1, output2]) #axis축
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
########### output 만들기 (분기)
output3 = Dense(30)(middle1)#middle1에서 받아온다 
output3 = Dense(7)(output3)
output3 = Dense(3)(output3)#outputs에 마지막 아웃풋 넣기 

output4 = Dense(15)(middle1)
output4_1 = Dense(14)(output4)
output4_2 = Dense(11)(output4_1)
output4_3 = Dense(3)(output4_2)#마지막 아웃풋 넣기 

#y3추가했을 때 ++++++++++++++++++++++++++++
output5 = Dense(30)(middle1)
output5_1 = Dense(20)(output5)
output5_2 = Dense(10)(output5_1)
output5_3 = Dense(3)(output5_2)#마지막 아웃풋 넣기 

#모델 정의하기 
model = Model (inputs =[input1, input2], outputs =[output3, output4_3, output5_3])
                        #모델 1,2넣기 
model.summary()

#concatenate는 연산하지 않고 병합만 해준다 

# 컴파일, 훈련 
model.compile(loss= 'mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y2_train, y2_train, y3_train], epochs=100, batch_size=8, validation_split=0.25, verbose=1)
                                                        #ㄴy3추가
# 평가 
result = model.evaluate([x1_test, x2_test],[y1_test,y2_test, y3_test], batch_size=8)
                                                                #ㄴy3추가
# (loss, 메트릭스의 mse)가 나온다 

print('result:' , result) #result: [61805.80859375, 61702.08203125, 59.0320930480957, 44.6912956237793, 203.48248291015625, 5.990623950958252, 4.886919021606445]

#result: [62924.65234375, 62796.45703125, 128.2028350830078, 204.76409912109375, 7.117029666900635] 총 5개 나옴 
#앙상블에서 output이 2개 이기 때문에 모델1, 모델2 각각 훈련한다. 
#첫번째값은 모델전체의 아웃풋(모델1아웃풋값+모델2아웃풋값의더한값), 모델1의 마지막아웃풋, 모델2의 마지막아웃풋, 모델1의메트릭스mse값, 모델2의 메트릭스 mse값
#앙상블을 3개 합치면 result값이 7개 