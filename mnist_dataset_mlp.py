#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl import NeuralNetwork 


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
    
    
    nn = NeuralNetwork([784,25,10], 2.5)
    
    plt.plot(nn.trainModel(np.array(x_train), y_train, mini_batch_size=64,epochs=30000, map_w=False))
   
    
    print(nn.evaluate(np.array(x_test), y_test))
