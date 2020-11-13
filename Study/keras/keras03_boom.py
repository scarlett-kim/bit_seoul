import numpy as np

#순서 1. x와 y 의 데이터 구성 

x = np.array([1,2,3,4,5,6,7,8,9,10]) #accuracy값을 1.0으로 올려보자. 
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

# 순서 2. 모델 구성
model = Sequential() 
model.add(Dense(300000, input_dim =1)) 
model.add(Dense(5000000))
model.add(Dense(300000))  
model.add(Dense(70000))
model.add(Dense(100000)) #데이터 양이 많으면 터짐. 


# 순서 3. 컴파일 , 화면 (수정가능하다 = 하이퍼 파라미터 튜닝)
model.compile(loss='mse', optimizer='adam', metrics=['acc']) 

model.fit(x, y, epochs=10000) #작업관리자에서 Gpu 사용하는 법 확인 cuda

# 순서 4. 평가, 예측
loss, acc = model.evaluate(x, y) 

print("loss: ", loss) #데이터 개수를 늘렸을 때 0.5이하로 나왔다 0.0008505168370902538
print("acc:" , acc) #0.10000000149011612

#예측 = 훈련시킨 값이 나온다 (accuracy 의 정확도를 위해서 예측이 필요하다)
y_pred = model.predict(x) #예측값을 y_pred라는 변수에 값을 넣는다 
print('결과물: \n :' , y_pred) #\n은 다음 줄





