#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from mpl import NeuralNetwork 





from mnist import MNIST

mnist = MNIST()
mnist.gz = True

x_train, y = mnist.load_training() #60000 samples
x_test, y_test = mnist.load_testing()    #10000 samples


y_train = np.zeros(len(y))

for i in range(len(y)):
    y_train[i] = y[i]
    

y_train = y_train.reshape(-1,1)



nn = NeuralNetwork([784,17,10], 0.1)




error = []
error.append(1)


for i in range(100000):
    print(i)
    idx = np.random.randint(0,len(x_train))

    #print("Treinando", x_train[idx][:4], np.array(y_train[idx]))
    aux = np.zeros(10)
    aux[int(y_train[idx])] = 1
    dw,db,e= nn.backprop(np.array(x_train[idx]), np.array(aux))
    error.append(e)
    dw = dw[0]        

    nn.w[0] = nn.w[0] - nn.learning_rate*np.array(dw)
    nn.b = nn.b - nn.learning_rate*np.array(db)
    



error_teste = []
for p,r in zip(x_train[:4],y_train[:4]):
    a = nn.predict(np.array(p))
    print("predição:", a)
    print("Res correto:", r)
    error_teste.append((a - r)**2)

plt.semilogx(error)
#plt.semilogx(error_teste)


