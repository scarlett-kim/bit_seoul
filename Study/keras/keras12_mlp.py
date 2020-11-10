# 데이터 shape
#multi layer 퍼섹트롬
import numpy as np

x = np.array([range(1,101), range(311,411), range(100)])
y = np.array([range(101,201), range(711, 811), range(100)])

print(x[1][10]) #100개의 데이터 3개 
print(x.shape) #(3,) 스칼라 30개 = 객체 inputing=1 / vector=1차원/ 매트릭스(행렬)=2차원 / 텐서 =3차원
#나는 shape가 (100,3) 이 나오길 바란다 

#(100,3) 형태로 x,y를 변환시켜라 

'''
주석 가능 
'''