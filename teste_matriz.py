# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:33:27 2020

@author: maruzka
"""

import numpy as np
from functools import reduce

layers = [4, 3, 1]
w = np.array([np.random.randn(j,i) for i,j in zip(layers[:-1], layers[1:])])
b = np.array([np.random.randn(i) for i in layers[1:]])




for i in w:
    print(i)
    
print("Multiplicacai")
print(np.dot(w[0], np.transpose([0,0,0,0])))


#def mdot(*args):
#    return reduce(np.dot, args)
#
#
#
#
#def output(x,w,b):
#    
#    return np.dot(w,output()) + b
#
#
#a = lambda x,w,b : (np.dot(w, x) + b)
#print(a(np.array([0,0,0,2]), w[0], b[0]))
#
#def mdt(*args):
#    return reduce(output, args)
#
#print(mdt(np.array([0.1,2,3,4]), w, b))


print(reduce(lambda a,d: a + d, [1,2,3], 1))

a = [1,2,3,4]
print(b)

