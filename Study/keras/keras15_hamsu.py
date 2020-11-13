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

# 모델 구성 
# from tensorflow.keras.models import Sequential #케라스 백엔드에 텐서플로우가 돌아간다. from keras.models import Sequential 도가능하지만 한 단계 더 거쳐서 더 느림
# from tensorflow.keras.layers import Dense

# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Input

# model = Sequential() #시퀀셜 모델 정의하기
# model.add(Dense(5, input_shape =(3, ), activation='relu')) #dense층에서는 명시해주지않으면 linear라는 활성화함수를 고정으로 쓰고 있다 
# model.add(Dense(4, activation='relu'))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(1))

# model.summary()#레이어와 파라미터 나옴 


#컴파일 훈련
# model.compile(loss='mse', optimizer='adam', metrics='mae')
# #model.fit(x_train, y_train, epochs=100, validation_split=0.25, verbose=2) #verbose라는 파라미터
# model.fit(x_train, y_train, epochs=100, validation_split=0.25, verbose=0) #verbose중간에 실행과정이 안 나온다 loss: 3858.9939 - mae: 53.5642
# model.fit(x_train, y_train, epochs=100, validation_split=0.25, verbose=1) 
# # loss: 506.1399 - mae: 19.6918 - val_loss: 1417.1885 - val_mae: 32.8744 Epoch 96/100
# model.fit(x_train, y_train, epochs=100, validation_split=0.25, verbose=3) 


# #평가, 예측
# loss,mse =model.evaluate(x_test, y_test)
# print(loss)
# print(mse)

# y_predict = model.predict(x_test)
# # print(y_predict)

# #RMSE
# #사이킷런 지표
# from sklearn.metrics import mean_squared_error
# # 사이킷런에 metrics mse 제공
# def RMSE(y_test, y_predict) :
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print('RMSE:' , RMSE(y_test, y_predict)) 

# #R2  
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print('R2:', r2) 






#########################################################################################

# ++ 함수형 모델 import하기 +++++++++ activation부분이 히든레이어? 
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Input
# #함수형 모델은 일반 모델 구성과 다르게 먼저 모델을 불러오고 나중에 모델을 정의한다 
# input1 = Input(shape=(3,)) #input레이어 구성한다 
# dense1 = Dense(10, activation='relu')(input1)#이렇게 (input1)을 가져오면 위에꺼를 사용하겠다 = input1 = Input(shape=(3,))
#  #레이어마다 활성화함수가 있고, 활성화함수 구성하자
# dense2 = Dense(5, activation='relu')(dense1)
# dense3 = Dense(7, activation='relu')(dense2)
# output1 = Dense(1)(dense3) # 왜 Dense(1)만 나왔을 까? 왜 activation표시 안 했을까? 디폴트값 linear, 마지막 activation은 linear여야 한다 
# model = Model(inputs = input1, outputs = output1) # 함수형 모델은 모델 정의를 마지막에 해준다 그리고 파라미터 안에 input1과 output1을 지정해준다 

## ++ 함수형 모델의 summary 보기 +++
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
#함수형 모델은 일반 모델 구성과 다르게 먼저 모델을 불러오고 나중에 모델을 정의한다 
input1 = Input(shape=(3,)) #input레이어 구성한다 
dense1 = Dense(5, activation='relu')(input1)#이렇게 (input1)을 가져오면 위에꺼를 사용하겠다 = input1 = Input(shape=(3,))
 #레이어마다 활성화함수가 있고, 활성화함수 구성하자
dense2 = Dense(4, activation='relu')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(1)(dense3) # 왜 Dense(1)만 나왔을 까? 왜 activation표시 안 했을까? 디폴트값 linear, 마지막 activation은 linear여야 한다 
model = Model(inputs = input1, outputs = output1) # 함수형 모델은 모델 정의를 마지막에 해준다 그리고 파라미터 안에 input1과 output1을 지정해준다 

model.summary()



'''

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
# print(y_predict)

#RMSE
#사이킷런 지표
from sklearn.metrics import mean_squared_error
# 사이킷런에 metrics mse 제공
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE:' , RMSE(y_test, y_predict)) 

#R2  
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2:', r2) 

'''


