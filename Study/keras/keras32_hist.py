import numpy as np
from tensorflow.keras.models import Sequential, load_model #load_model에는 Sequential까지 포함되어있음 
from tensorflow.keras.layers import Dense, LSTM

# 데이터 keras 26번 파일 split1 
dataset = np.array(range(1,101))

size=5 #사이즈의 크기만큼 짤라준다 

def split_x(seq, size) : 
    aaa=[]
    for i in range(len(seq)-size+1) :
        subset=seq[i:(i+size)]
        aaa.append([item for item in subset]) #aaa.append뒤에 for을 꼭 넣지 않아도 된다  
    print("값 ", type(aaa)) # <class 'list'>
    return np.array(aaa)

datasets = split_x(dataset, size)

x = datasets [:, 0 : 4] # 모든 행 콤마 4번째에 있는 열까지 
y = datasets [:, 4]

x = np.reshape (x, (x.shape[0], x.shape[1], 1 ))
# x = np.reshape (x, (x.shape[0], x.shape[1], 1 , 1 )) 4차원
# x = np.reshape (x, (x.shape[0], x.shape[1], 1 , 1, 1)) 5차원
print(x.shape) 
# 값  <class 'list'>
# (96, 4, 1)


# train, test 분리 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7)

print(x_train.shape) #3차원넘어서 4, 5차원도 다 스플릿 가능 (67, 4, 1)

# 모델 구성 
# model = Sequential()
# model.add (LSTM (100, input_shape = (4,1)))
# model.add (Dense(50))
# model.add (Dense(10))
# model.add (Dense(1))

# 모델 불러오기 load_model import하기 / 
model = load_model ('./save/keras.28.h5') #모델이 좋다면 저장

# 위 모델에서 + 플러스해서 모델을 더 추가하고 싶다 
model.add(Dense(5, name ='king'))
model.add(Dense(1, name ='king2'))
# ValueError: All layers added to a Sequential model should have unique names.
# Name "dense" is already the name of a layer in this model. Update the `name` argument to pass a unique name.
model.summary()

# 컴파일, 훈련 
from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss ='mse', optimizer='adam', metrics=['mae'])
early_stopping=EarlyStopping(monitor='loss', patience=50, mode='auto')
history = model.fit(x_train, y_train, epochs= 100, batch_size =1, verbose =1, validation_split =0.2, callbacks=[early_stopping])

print("==================")
print(history)
print("===================")
print(history.history.keys()) # dict_keys(['loss', 'mse', 'val_loss','val_mse'])
print("===================")
print(history.history['loss']) # loss수치 
print("===================")
print(history.history['val_loss']) # validation수치 
print("===================")
# model.fit (dataset_train, epochs=100, batch_size=8, validation_split=0.8, verbose=1, callbacks=[early_stopping])

# # 평가 
# result =model.evaluate(dataset_test)
x_input=np.array([5,6,7]) #(3,) ->(1,3,1) 입력에 형변환이 있었으므로 예측값을 내는 자료도 형변환을 해줘야 함. [5,6,7]을 [[5], [6], [7]] 형태로....
x_input= x_input.reshape(1,3,1)

# loss=model.evaluate(x_test, y_test, batch_size=1)

y_predict=model.predict(x_input)

# print("loss : ", loss)
print("y_predict : ", y_predict)
