
import numpy as np
# from keras25_split import split_x #25번 파일의 함수 split_x를 불러온다 

dataset = np.array(range(1,11))# 1부터 11까지
size = 5 #사이즈는 5
 
def split_x(seq, size) : 
    aaa=[]
    for i in range(len(seq)-size+1) :
        subset=seq[i:(i+size)]
        aaa.append([item for item in subset]) #aaa.append뒤에 for을 꼭 넣지 않아도 된다  
    print("값 ", type(aaa)) # <class 'list'>
    return np.array(aaa)

datasets=split_x(dataset, size)

print('======================')
print("datasets", datasets)

'''
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]
에 데이터 5, 6, 7, 8, 9, 10 모든 행마다 마지막 숫자가 y값으로 뽑아낸다 
결국 최종적으로 [7 8 9 10 11] 이라는 값에서 y값 =11이라는 값을 추출해야 한다 '''

# datasets = datasets.reshape (6, 4, 1)
# print("x.reshape 값은? " , datasets)

# 슬라이싱하기 
x_train = datasets[:, :size-1] #첫 째줄 [1 2 3 4 ]
y_train = datasets[:, size-1:] #첫 째줄 [5] y값 
# x_test = datasets[80:] 
# y_test = datasets[20:]
print("x_train", x_train) 
# x_train 
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]]
print("y_train", y_train)
# y_train
#  [[ 5]
#  [ 6]
#  [ 7]
#  [ 8]
#  [ 9]
#  [10]]

# x_train.reshape = x_train.reshape(6,4,1) 하나씩 나누기 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1) 

print('reshape하고 난 뒤 x값', x_train)
# reshape하고 난 뒤 x값
#  [[[1]
#   [2]
#   [3]
#   [4]]

#  [[2]
#   [3]
#   [4]
#   [5]]

#  [[3]
#   [4]
#   [5]
#   [6]]

#  [[4]
#   [5]
#   [6]
#   [7]]

#  [[5]
#   [6]
#   [7]
#   [8]]

#  [[6]
#   [7]
#   [8]
#   [9]]]
print('reshape하고 난 뒤 y값', y_train)


#슬라이싱 train_test_split 
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x_train, y_train, shuffle=False, train_size =0.7)
print('뭘까', x_train) 


# 모델을 정의하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model=Sequential()
model.add(LSTM(20, activation='relu', input_shape=(4,1)))   
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(1))

model.summary()

# 컴파일, 훈련 
model.compile(loss ='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping

early_stopping=EarlyStopping(monitor='loss', patience=50, mode='auto')
# model.fit (dataset_train, epochs=100, batch_size=8, validation_split=0.8, verbose=1, callbacks=[early_stopping])

# # 평가 
# result =model.evaluate(dataset_test)
x_input=np.array([5,6,7]) #(3,) ->(1,3,1) 입력에 형변환이 있었으므로 예측값을 내는 자료도 형변환을 해줘야 함. [5,6,7]을 [[5], [6], [7]] 형태로....
x_input= x_input.reshape(1,3,1)

# loss=model.evaluate(x_test, y_test, batch_size=1)

y_predict=model.predict(x_input)

# print("loss : ", loss)
print("y_predict : ", y_predict)
