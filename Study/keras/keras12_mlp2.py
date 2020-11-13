# 데이터 shape
#multi layer 퍼섹트롬
import numpy as np

x = np.array([range(1,101), range(711,811), range(100)])
y = np.array(range(101,201))

print(x) 
print("transpose하기전" , x.shape) #transpose하기전 (3, 100)
print("t" , y.shape) #(100, )

x= np.transpose(x)
y= np.transpose(y)
print("transpose 하고 난 후" ,x.shape) #transpose 하고 난 후 (100, 3)


#모델 구성하기 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim =3)) # 3으로 맞춰주고
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1)) #1로 맞춰주고 


#나머지 완성 


