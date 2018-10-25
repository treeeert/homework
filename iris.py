import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv('D:/python34/machine learning/iris.csv')

train.loc[train.Species == 'setosa','Species'] = 0 #将setosa类归为0
train.loc[train.Species == 'versicolor','Species'] = 1 #将veisicolor类归为1
y = train.Species
train_X = train.drop(['Species'],axis = 1)

def Sigmoid(inX):
    return 1/(1+np.exp(-inX))  #sigmoid函数

def Training(data,label):
    dataMatrix = np.mat(data).astype(np.float64)
    labelMatrix = np.mat(label).transpose().astype(np.float64)#将输入输出转化为矩阵形式
    dataMatrix = dataMatrix[:,1:6]#取输入矩阵的2~5列
    m,n = np.shape(dataMatrix)
    a = np.ones((m,1))
    dataMatrix = np.c_[a,dataMatrix]   #为输入矩阵增添一个全为1的列作为其第一列
    w = np.ones((n+1,1))  #初始参数全部置为1
    alpha = 0.01     #取学习率，即下降步长
    c = []
    for i in range(1000):
        prediction = Sigmoid(dataMatrix * w)
        loss = prediction - labelMatrix
        d = np.ones((1,m))
        c.append(np.ravel(d*loss))  #保存每次学习的损失
        w = w - alpha * dataMatrix.transpose() * loss   #梯度下降更新参数
    return w,c


x,y = Training(train_X,y)
print(x)
print(y)
plt.plot(y)   #得到迭代时损失变化的曲线
plt.show()
