import numpy as np

#순서 1. x와 y 의 데이터 구성 

x = np.array([1,2,3,4,5,]) #정제된 x, y 1차원 데이터를 만든다 
y = np.array([1,2,3,4,5,])

from tensorflow.keras.models import Sequential #텐서플로우안에 케라스안에 시퀀셜(순차적으로)을 가져오겠다
from tensorflow.keras.layers import Dense #딥러닝에서 레이어의 dense층을 쓰겠다

# 순서 2. 모델 구성 (레이어의 깊이와 노드의 개수 수정가능하다 = 하이퍼 파라미터 튜닝))
#메모장에 그린 딥러닝 (눈과 입 사이 신경망 구조) 하나하나가 y = wx+ b 구조다 
#dense = dnn 이다 
#2-1.기초적인 모델 시퀀셜
#model = Sequential() #시퀀셜모델을 모델이라고 부르겠다.
#model.add(Dense(3, input_dim =1)) #덴스층을 add 3개의 층를 쌓겠다 dimention ? 디멘션
#model.add(Dense(5))
#model.add(Dense(3)) #또 층을 쌓겠다 
#model.add(Dense(1)) 
# #순차적으로 가야지만 딥러닝 신경망을 갈 수 있기 때문에 model=Sequential로 정의해야한다 
# 위에 데이터를 1차원을 넣었기 때문에 1 만약에 x = [1,2] , [3,4]로하면 2

#2-2.숫자를 늘렸을 때 시퀀셜 오케이 (하이퍼 파라미터 튜닝? 숫자변경, 파라미터 추가 등)
#model = Sequential() 
#model.add(Dense(300, input_dim =1)) 
#model.add(Dense(5000))
#model.add(Dense(30))  
#model.add(Dense(1))

#2-3.레이어의 깊이를 늘려보자 그래도 오케이 
model = Sequential() 
model.add(Dense(300, input_dim =1)) 
model.add(Dense(5000))
model.add(Dense(30))  
model.add(Dense(7))
model.add(Dense(1))


# 순서 3. 컴파일 , 화면 (수정가능하다 = 하이퍼 파라미터 튜닝)
model.compile(loss='mse', optimizer='adam', metrics=['acc']) 
#loss 값을 mse = mean squared error 손실함수, 손실 오차 / mse로 잡겠다 
##optimizer를 아담으로 쓰겠다 
#눈으로 보는 평가지표는 metrics = accuracy 정확도를 사용하겠다 

#머신을 훈련시킨다 (헬스장가서 훈련.. ㅋㅋfit)
#model.fit(x, y, epochs=100, batch_size=1) #batch_size =1이면 1개씩 잘라서 sport2=100개를 넣겠다 
model.fit(x, y, epochs=100) #batch_size디폴트 값이 32이기 때문에 오류가 없다

# 순서 4. 평가=> 평가가 나오면 결과를 loss와 acc로 반환한다 (수정가능하다 = 하이퍼 파라미터 튜닝)
#loss, acc = model.evaluate(x, y, batch_size=1) 
loss, acc = model.evaluate(x, y) #batch_size디폴트 값이 32이기 때문에 오류가 없다

print("loss: ", loss) #0.021037546917796135 loss는 낮추고
print("acc:" , acc) #0.20000000298023224 acc를 1.0까지 올려라

#머신러닝을 돌리다가 crtl+ c하며 중단된다 

# + 훈련된 것으로 평가했기 때문에  모의고사를 보고 그 다음날 수능문제가 모의고사 문제와 같다면 ? 
# + accuracy가 정확도가 떨어진다 
#=> 이 2가지를 업그레이드 하려면 ?  이제부터 2번째 강의에서 시작! 


