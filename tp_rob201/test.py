import numpy as np


#print(np.arccos(-1))
v1 = [1,2,3]
v2 = [np.pi,0,np.pi/2]

v3 = v1*np.cos(v2)
# print(v3)

v4 = np.array([1,2,1])
# print(v4.shape[1])

v5 = v1[0:2]


reserve = v4 < v4.max()
v4_select = v4[reserve]

sigma = [1, 0.2, 0.2] 
offset = np.random.normal([1,3,6], sigma)

offset = np.random.normal(0,0.2,size=2)
print(offset)