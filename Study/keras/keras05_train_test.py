import numpy as np

#순서 1. x와 y 의 데이터 구성 
#학습 시킨 데이터로 모델링을 했기 때문에 , 정답이 있는 문제나 다름없다 
x = np.array([1,2,3,4,5,6,7,8,9,10]) #accuracy값을 1.0으로 올려보자. 
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_pred = np.array([11,12,13])#그래서 데이터를 추가한다 , 학습 시킨 데이터로 모델링을 했기 때문에 , 정답이 있는 문제나 다름없다 

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

# 순서 2. 모델 구성 
model = Sequential() 
model.add(Dense(3, input_dim =1)) #레이어 개수 5개
model.add(Dense(50))
model.add(Dense(30))  
model.add(Dense(7))
model.add(Dense(1)) # 노드 개수 수정.



# 순서 3. 컴파일 , 화면 (수정가능하다 = 하이퍼 파라미터 튜닝)
# model.compile(loss='mse', optimizer='adam', metrics=['mse']) loss:  [1.324835466220975e-06, 1.324835466220975e-06]
# model.compile(loss='mse', optimizer='adam', metrics=['mae']) #mae를 넣으면 loss값을 mae로 판단한다 loss:  [2.3447910280083306e-13, 3.2186508747145126e-07]
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 

#model.compile(loss='mse', optimizer='adam', metrics=['acc']) 
#ㄴmatrics가 있을 땐, loss,acc값이 나온다 
# model.compile(loss='mse', optimizer='adam') 
#ㄴmatrics을 제외하면 loss값만 나옴 

model.fit(x, y, epochs=10000) 

# 순서 4. 평가, 예측
#loss, acc = model.evaluate(x, y) 
loss = model.evaluate(x, y)  
# =>매트릭스['acc']있고 loss값만 반환할 경우에도 acc값이 나오고 loss값은 [0.2222, 0.3333] 리스트반환
# 매트릭스['mse']라면? loss:  [1.324835466220975e-06, 1.324835466220975e-06]

print("loss: ", loss) 
# print("acc:" , acc) 


#예측 = 훈련시킨 값이 나온다 (accuracy 의 정확도를 위해서 예측이 필요하다)
y_pred = model.predict(x_pred) #x_pred라는 변수에 데이터를 11,12,13을넣고 다시 예측하기 
print('결과물: \n :' , y_pred) #결과물:: [[10.999997]  [12.   정수   ] [13.      ]]




