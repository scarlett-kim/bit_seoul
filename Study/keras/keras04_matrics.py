import numpy as np

#순서 1. x와 y 의 데이터 구성 

x = np.array([1,2,3,4,5,6,7,8,9,10]) #accuracy값을 1.0으로 올려보자. 
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

# 순서 2. 모델 구성
model = Sequential() 
model.add(Dense(3, input_dim =1)) 
model.add(Dense(50))
model.add(Dense(30))  
model.add(Dense(7))
model.add(Dense(1)) # 노드 개수 수정.


# 순서 3. 컴파일 , 화면 (수정가능하다 = 하이퍼 파라미터 튜닝)
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])
#model.compile(loss='mse', optimizer='adam', metrics=['acc']) 
#ㄴmetrics가 있을 땐, loss,acc값이 나온다 
model.compile(loss='mse', optimizer='adam') 
#ㄴmetrics을 제외하면 loss값만 나옴 

model.fit(x, y, epochs=10000) 

# 순서 4. 평가, 예측
#loss, acc = model.evaluate(x, y) 
loss = model.evaluate(x, y)  
# =>매트릭스가있고 loss값만 반환할 경우에도 acc값이 나오고 loss값은 [0.2222, 0.3333] 리스트반환

print("loss: ", loss) 
# print("acc:" , acc) 


#예측 = 훈련시킨 값이 나온다 (accuracy 의 정확도를 위해서 예측이 필요하다)
y_pred = model.predict(x) 
print('결과물: \n :' , y_pred) 






