# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:37:42 2020

@author: maruzka
"""

import numpy as np
import matplotlib.pyplot as plt

class CrossEntropyCost(object):

        @staticmethod
        def fn(a, y):
            """Return the cost associated with an output ``a`` and desired output
            ``y``.  Note that np.nan_to_num is used to ensure numerical
            stability.  In particular, if both ``a`` and ``y`` have a 1.0
            in the same slot, then the expression (1-y)*np.log(1-a)
            returns nan.  The np.nan_to_num ensures that that is converted
            to the correct value (0.0).
            """
            return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
        @staticmethod
        def delta(z, a, y):
            """Return the error delta from the output layer.  Note that the
            parameter ``z`` is not used by the method.  It is included in
            the method's parameters in order to make the interface
            consistent with the delta method for other cost classes.
            """
            return (a-y)


class NeuralNetwork:
    
    
    def __init__(self, layers, learning_rate, cost_function = CrossEntropyCost):
        
        self.layers = layers
        self.num_layers = len(layers)
          
        
        self.b = np.array([np.random.randn(i,1) for i in layers[1:]])
        self.w = np.array([np.random.randn(j,i) for i,j in zip(layers[:-1], layers[1:])])
    
    
        self.cost = cost_function
#        mamae = [np.array([[0.15, 0.2],[0.25, 0.3]]), np.array([[0.4, 0.45],[0.5, 0.55]])]
#        doceu = [ np.array([[0.35],[0.35]]), np.array([[0.6],[ 0.6]])]
#
#        for i in range(len(self.w)):
#            self.w[i] = mamae[i]
#            
#        for j in range(len(self.b)):
#            self.b[j] = doceu[j]
#            

        self.learning_rate = learning_rate

        
    def loadModel():
        # TODO 
        return 1
     
    def saveModel():
        # utilziar cpickle? 
        return 1
    
    def backprop(self, x, y):
        
  
        act, z = self.feedfoward(x)
        
        dB = [np.zeros(b.shape) for b in self.b]
        dW = [np.zeros(w.shape) for w in self.w]
        
        g = act[-1] - y
        
        e = np.sum((y-act[-1])**2)
        
        assert g.shape == act[-1].shape
        
        for l in range(1, self.num_layers):
            #g = g
            
#            if l > 1:
            g = g * self.dsigmoid(z[-l])
            
                
            try:
    
                aux = [i*act[-l-1] for i in g]
                dW[-l] = aux
            
            except ValueError:
               # print("mds n guento mais")
                dW[-l] = g * act[-l-1].T
                
            except TypeError:
                print("TyperError")

                
            dB[-l] = g

            g = np.dot(self.w[-l].T, g)
            
            
        return [dW,dB,e]
        #return (np.array([1,1]), np.array([2,2]), 9)
        
        
        
    def feedfoward(self, x):
        
        
        activation = x
        activations = [x] # list to store all the activations, layer by layer
       
        zs = [] # list to store all the z vectors, layer by layer
    
        for b, w in zip(self.b, self.w):
            
            z = np.dot(w, activation) + b.T[0]
            zs.append(z)
            
            activation = self.sigmoid(z)
            activations.append(activation)

        return np.array(activations),np.array(zs)
    
    def predict(self, x):
        
        """
        TODO: Pensar em multi_dot numpy 
        
        b.T[0] -> é feito para ajustar o formato das arrays para que 
        a soma possa ser realizada corretamente
        
        por ex: b = [[a], b.T -> [[a, b]] -> b.T[0] -> [a, b]
                     [b]]
        
        dessa forma o resultado de x = self.sigmoid(np.dot(w, x) + b.T[0])
        mantém a estrutura [y0, y1, y2]

        """
        
#        print("Predição")
        for b, w in zip(self.b, self.w):
       
            x = self.sigmoid(np.dot(w, x) + b.T[0])

        
        return x
#
    
    
    
    @staticmethod
    def mini_batch2(x, y, dataset_lenght, mini_batch_lenght):
    
        #idx = np.random.randint(0,dataset_lenght-mini_batch_lenght)
        idx = np.random.randint(0, dataset_lenght, mini_batch_lenght, dtype = 'I')

        return (x[idx],y[idx])


    
    
    @staticmethod
    def mini_batch(x, y, dataset_lenght, mini_batch_lenght, idx):
    
        #idx = np.random.randint(0,dataset_lenght-mini_batch_lenght)
        #idx = np.random.randint(0, dataset_lenght, mini_batch_lenght, dtype = 'I')

        return (x[idx:idx+mini_batch_lenght],y[idx:idx+mini_batch_lenght])


    def stochastic_gradient_descent(self, x, y):
    
    
        db = np.array([np.zeros(b.shape) for b in self.b])
        dw = np.array([np.zeros(w.shape) for w in self.w])
        err = []    
        
        for i,j in zip(x,y):
            
            w,b,e = self.backprop(i,j)
            db += b
            dw += w
            err.append(e)
        
        db = db / len(x)
        dw = dw / len(x)
        
        err = np.sum(err)/len(x)
        
       # print("er", err)
        
        return dw,db,err
    
    
    
    def trainModel(self, training_data, desired_response, epochs, mini_batch_size = 60000):
        
        j = 0
        error = []
        
        while j < epochs:
            print("epoch: ",j+1)
            
#            x,y = self.mini_batch2(training_data, desired_response, dataset_lenght=len(training_data), mini_batch_lenght=mini_batch_size)
           
            # ruim em termos de mem
            mini_batches = [
                    self.mini_batch(training_data, desired_response, dataset_lenght=len(training_data), mini_batch_lenght=mini_batch_size, idx=idx) 
                    for idx in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
            #for i in range(0, len(training_data), mini_batch_size):
                
#                dw,db,e = self.stochastic_gradient_descent(training_data[i:i+mini_batch_size], desired_response[i:i+mini_batch_size])
                dw,db,e = self.stochastic_gradient_descent(mini_batch[0], mini_batch[1])
#            dw,db,e = self.stochastic_gradient_descent(x, y)

                error.append(e)
            
                dw = dw[0]        
            
                # Atualiza os pesos e bias utilizando os gradiente
                self.w[0] = self.w[0] - self.learning_rate*np.array(dw)
                self.b = self.b - self.learning_rate*np.array(db)
            
            j+=1
            
        return error
    
    
    def evaluate(self, x_test, y_test):
        
        results = [(np.argmax(self.predict(x)), np.argmax(y))
                       for (x, y) in zip(x_test, y_test)]

        return sum(int(x == y) for (x, y) in results)
        
        
    
    @staticmethod
    def sigmoid(x):
                
        return 1.0 / (1.0 + np.exp(-x));

    @staticmethod
    def dsigmoid(x):
        
        return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x));
        
        


if __name__ == "__main__":
    
    np.random.seed(0)
    nn = NeuralNetwork([2,4,1], 0.5)
    
#    
    dataset = np.array([[1,0],
                        [0,1],
                        [0,0],
                        [1,1]])
    
    dataset2 = np.array([[0],
                         [1],
                         [0],
                         [1]])
            

    plt.plot(nn.trainModel(dataset, dataset2, epochs=5000, mini_batch_size = 2))
    
    plt.legend(["1","2","4"])

    for i,j in zip(dataset, dataset2):
        print("Predicao: ", nn.predict(i))
      

        
    
    

#    