import numpy as np

#순서 1. x와 y 의 데이터 구성 

x = np.array([1,2,3,4,5,6,7,8,9,10]) #accuracy값을 1.0으로 올려보자. 
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

# 순서 2. 모델 구성
model = Sequential() 
model.add(Dense(300, input_dim =1)) 
model.add(Dense(5000))
model.add(Dense(30))  
model.add(Dense(7))
model.add(Dense(1))


# 순서 3. 컴파일 , 화면 (수정가능하다 = 하이퍼 파라미터 튜닝)
model.compile(loss='mse', optimizer='adam', metrics=['acc']) 

model.fit(x, y, epochs=100) 

# 순서 4. 평가, 예측
loss, acc = model.evaluate(x, y) 

print("loss: ", loss) #데이터 개수를 늘렸을 때 0.5이하로 나왔다 0.0008505168370902538
print("acc:" , acc) #0.10000000149011612

#예측 = 훈련시킨 값이 나온다 (accuracy 의 정확도를 위해서 예측이 필요하다)
y_pred = model.predict(x) #예측값을 y_pred라는 변수에 값을 넣는다 
print('결과물: \n :' , y_pred) #\n은 다음 줄

#결과물: 이 평가 예측이 맞지 않다. 1과 0.9는 같지 않다 
# : [[ 0.9665452]
#[ 1.9725055]
#[ 2.978466 ]
#[ 3.9844265]
#[ 4.990387 ]
# [ 5.9963474]
#[ 7.0023074]
#[ 8.008268 ]
#[ 9.014229 ]
#  [10.020188 ]]

#1. 선형회귀기법(regress) : accuracy라는 방식을 쓸 수 없다 / 실수값을 쓸 때 쓰는 기법
#2. 분류기법(classfier): 한다 / 안한다 = 결과값이 2개만 존재하면 된다 , 결과값이 0이냐 1이냐만 나오면 된다 ,동전의 앞과 뒤
# 먼저 모델 기법이 먼저 정하고 모델링을 한다 ex)꽃 3개가 있다면, 장미, 모닝글로리, 튤립 중에 수치가 가장 높은 것을 찾고
#그걸을 분류기법으로 쓰면 수치가 아닌 장미 라고 나온다 





