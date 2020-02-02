#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from mpl import NeuralNetwork 

#
#
#error = []
#error.append(1)
#
#
#def mini_batch(x, y, dataset_lenght, mini_batch_lenght):
#    
#    #idx = np.random.randint(0,mini_batch_lenght)
#    idx = np.random.randint(0, dataset_lenght, mini_batch_lenght)
#    
#    return x[idx],y[idx]
#    
#    
#def stochastic_gradient_descent(x, y):
#    
#    
#    db = np.array([np.zeros(b.shape) for b in nn.b])
#    dw = np.array([np.zeros(w.shape) for w in nn.w])
#    err = []    
#    
#    for i,j in zip(x,y):
#        
#        w,b,e = nn.backprop(i,j)
#        db += b
#        dw += w
#        err.append(e)
#    
#    db = db / len(x)
#    dw = dw / len(x)
#    err = np.sum(err)/len(x)
#    
#    
#    return dw,db,err
#    
#
#
#x_train = np.array([[0,0],
#                        [0,1],
#                        [1,0],
#                        [1,1]])
#    
#y_train = np.array([[0],
#                         [1],
#                         [1],
#                         [1]])
#
#
#
#for i in range(20000):
#    print(i)
#
#    x,y = mini_batch(x_train, y_train, dataset_lenght=len(x_train), mini_batch_lenght=3)
#    dw,db,e = stochastic_gradient_descent(x,y)
#    error.append(e)
#    dw = dw[0]        
#
#    nn.w[0] = nn.w[0] - nn.learning_rate*np.array(dw)
#    nn.b = nn.b - nn.learning_rate*np.array(db)
#    
#
#
#
#error_teste = []
#for p,r in zip(x_train[:4],y_train[:4]):
#    a = nn.predict(np.array(p))
#    print("predição:", a)
#    print("Res correto:", r)
#    error_teste.append((a - r)**2)
#
#plt.plot(error)
#plt.plot(error_teste)


if __name__ == "__main__":
    
    from mnist import MNIST

    mnist = MNIST()
    mnist.gz = True
    
    
    np.random.seed(0)
    
    x_train, y = mnist.load_training() #60000 samples
    x_test, y_t = mnist.load_testing()    #10000 samples
    
    
    y_train = np.zeros([len(y), 10])
    for i in range(len(y_train)):
        y_train[i][int(y[i])] = 1
        
    
    y_test = np.zeros([len(y_t), 10])
    for i in range(len(y_t)):
        y_test[i][int(y_t[i])] = 1
    
    
    nn = NeuralNetwork([784,25,10], 0.3)
    
    plt.plot(nn.trainModel(np.array(x_train), y_train, mini_batch_size=5,epochs=5))
   
    
    print(nn.evaluate(np.array(x_test), y_test))
