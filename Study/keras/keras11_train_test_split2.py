#1. 데이터 valication 으로 검증데이터 할 때 슬라이싱 없이 하기 위해 +validation_data를 써서 조작해봐라,
#슬라이싱은 섞이지 않아서, train_test_split 
import numpy as np
x = np.array(range(1,101)) #weight =1, 바이어스=100 , 1~100까지
y = np.array(range(101,201)) # 101~200까지

# 데이터 슬라이싱 없이
#사이킷런 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size =0.7, shuffle=False)
#x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size =0.3, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size =0.2, shuffle=False) 

# shuffle =섞여있는 것이 True , defalut값이다
# shuffle = False x_test: [ 71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88
# 89  90  91  92  93  94  95  96  97  98  99 100]
# trainsize를 70%로 주겠다 그럼 test파일은 30$ // test_size로도 가능하다 

print("x_test:", x_test)#x_test: [ 93(이런객체를 스칼라라고칭한다)  28  21  45  99   6  19  64  40  18  94  43  65  67  15  44  47  29
# 92  69  14  57  79  46 100  32   9  35  85  63]  x가93이라면 y는 100차이라서 193 

print('x_test.shape', x_test.shape) #x_test.shape (30,) 스칼라 30개 = 객체 inputing=1 / vector=1차원/ 행렬=2차원 / 텐서 =3차원

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

model = Sequential() 
model.add(Dense(3, input_dim =1)) 
model.add(Dense(50))
model.add(Dense(30))  
model.add(Dense(7))
model.add(Dense(1))

# 순서 3. 컴파일 , 화면 
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 

model.fit(x_train, y_train, epochs=100 , validation_data=(x_val, y_val))

# 순서 4. 평가, 예측
#loss, acc = model.evaluate(x, y) 
loss = model.evaluate(x_test, y_test)  


print("loss: ", loss) 

#예측 = 훈련시킨 값이 나온다 (accuracy 의 정확도를 위해서 예측이 필요하다)
y_predict = model.predict(x_test) #x_test를 넣는 이유는 지표를 찾기 위해서 원래는 y_predict = model.predict(x_pred)

print('결과물: \n :' , y_predict) 

#RMSE는 파라미터말고 아래 추가해야한다
#사이킷런 지표
from sklearn.metrics import mean_squared_error
# 사이킷런에 metrics mse 제공
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE:' , RMSE(y_test, y_predict)) #0.20419436496650925

#R2 , accuracy대용으로 쓸 수 있다 RMSE랑 같이 써야 좋다 서로 보완하는 보완지표 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2:', r2) #R2: 0.979152330657962