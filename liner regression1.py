import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
###采样点(Xi,Yi)，一会加###
Xi = np.array([1,2,3,4,5,6,7])
Yi = np.array([1,3,5,8,9,12,16])







###需要拟合的误差函数func以及误差error###
def func(p,x):
    k,b = p
    return k*x+b
def error(p,x,y,s):
    print(s)
    return func(p,x)-y


p0 = [100,2]



###主函数由此开始###
s = 'Test the number of iteration'
Para = leastsq(error,p0,args = (Xi,Yi,s))
k,b = Para[0]
print('k=',k,'\n','b=',b)

###绘图，观察拟合效果###
plt.figure(figsize=(8,6))#指定图像比例8:6
plt.scatter(Xi,Yi,color = 'red',label = 'Sample Point',linewidth = 3) #画样本点
x = np.linspace(0,10,1000)
y=k*x+b
plt.plot(x,y,color = 'orange',label = 'Fitting Line',linewidth = 2) #画拟合直线
plt.legend()
plt.show()
###leastsq(func,x0,arg())函数一般指定前三个参数,func是误差函数，x0是计算的初始参数值，arg()是指定函数的其他参数值
