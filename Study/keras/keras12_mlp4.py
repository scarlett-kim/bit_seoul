# 데이터 shape
# 실습 train_test_split를 슬라이싱으로 바꾸기
import numpy as np

x = np.array([range(1,101), range(311,411), range(100)])
y = np.array([range(101,201), range(711, 811), range(100)])

print(x) #100개의 데이터 3개 
print("transpose하기전" , x.shape)#(3,10) #(3,) 스칼라 30개 = 객체 inputing=1 / vector=1차원/ 매트릭스(행렬)=2차원 / 텐서 =3차원
#나는 shape가 (100,3) 이 나오길 바란다 

#(100,3) 형태로 x,y를 변환시켜라 

#행무시, 열우선


x= np.transpose(x)
y= np.transpose(y)
print("transpose 하고 난 후" ,x.shape) #(100,3)

#y1, y2, y3 = w1*1 + w2*2 + w3*3 +b
#모델 구성하기 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim =3))
model.add(Dense(5))
model.add(Dense(3)) #출력 컬럼 3개 

