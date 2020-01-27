# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:37:42 2020

@author: maruzka
"""

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    
    def __init__(self, layers, learning_rate, activation_function = None):
        
        self.layers = layers
        self.num_layers = len(layers)
        
#        self.b = np.array([np.random.randint(1,3,[i,1]) for i in layers[1:]])
#        self.w = np.array([np.random.randint(1,9,[j,i]) for i,j in zip(layers[:-1], layers[1:])])   
        
        self.b = np.array([np.random.randn(i,1) for i in layers[1:]])
        self.w = np.array([np.random.randn(j,i) for i,j in zip(layers[:-1], layers[1:])])
    
#        mamae = [np.array([[0.15, 0.2],[0.25, 0.3]]), np.array([[0.4, 0.45],[0.5, 0.55]])]
#        doceu = [ np.array([[0.35],[0.35]]), np.array([[0.6],[ 0.6]])]
#
#        for i in range(len(self.w)):
#            self.w[i] = mamae[i]
#            
#        for j in range(len(self.b)):
#            self.b[j] = doceu[j]
#            
#              
#        print("Bias")
#        for i in self.b:
#            print(i)
#            
#        print("Weights")
#        for i in self.w:
#            print(i)
#        self.b = np.array([ np.array([0.35, 0.35]), np.array([0.6, 0.6])])
#        self.w = [np.array([[0.15, 0.2],[0.25, 0.3]]), np.array([[0.4, 0.45],[0.5, 0.55]])]
#        
        self.learning_rate = learning_rate
        
        # TODO: Criar uma maneira para que o usuario possa fornecer uma função de ativação de sua escolha
        """
        
        if activation_function == None:
            self.activation_function = self.sigmoid
        else:
            self.activation_function = self.sigmoid
        
        """
        
    def loadModel():
        # TODO 
        return 1
     
    def saveModel():
        # utilziar cpickle? 
        return 1
        
    def trainModel(self, training_data, desired_response, epochs):
        
        j = 0
        
        while j < epochs:
            print("epoch: ",j+1)
            #idx = np.random.randint(len(training_data))
            
            #print(training_data[idx], desired_response[idx])
            
            b,w = self.backpropagation(training_data, desired_response)
            
         
         
            print("atualizando pesos")
            print(self.w)
            print(w)
            for i in range(1,self.num_layers):
#                print("selfb", self.b[-i])
                print("Layers ", i)
                #print("b", b[-i])
                
                print("w", w[-i])
                self.w[-i] = self.w[-i] - self.learning_rate*(w[-i])
                #self.b[-i] = self.b[-i] - self.learning_rate*(b[-i])
                
                
                
                print("Pesos")
                for x in self.w:
                    print(x)
                
                assert (len(self.predict([0.1, 0.1])) == self.layers[-1])


        
            
            
  
            j+=1
        
                    


        #plt.plot((error)


    
    def backpropagation(self, x, y):
        """
        Referências: 
            https://sudeepraja.github.io/Neural/
            http://neuralnetworksanddeeplearning.com/chap2.html
            
            Para atualizar W é preciso transformar delta_w em um vetor coluna
            
        """
        activations, zs = self.feedfoward(x)
        
        print(activations)
        print(zs)
        # cria um vetor que recebera cada delta_b e delta_w associado ao peso e vies correspondente
        
        delta_b = [np.zeros(b.shape) for b in self.b]
        delta_w = [np.zeros(w.shape) for w in self.w]
        
        print("\nInit Backpropagation\n")
        
        print("Camada de saída\n")
        #print(activations[-1].reshape(-1,1) - y.reshape(-1,1)) 
        # Pode-se fazer reshape(-1,1) apenas no resultado da derivada
        
        delta = ((activations[-1]-y)*(self.dsigmoid(zs[-1])))
        
#        print("delta")
#        print(delta)
#        print("act")
#        print(activations[-2])
        
        delta_b[-1] = delta
        delta_w[-1] = (delta * activations[-2]).reshape(-1,1)
        
#        delta_w[-1] = activations[-2] * delta
        
#        print("delta_w\n",0.5*delta_w[-1])
#        
#        print("Corrigindo:\n",self.w[-1])
#        print("")

        print("\nCamandas ocultas\n")
        
        for l in range(2, self.num_layers):
            
            #print("Delta")
            a = np.dot(self.w[-l+1].T, delta)
            b = self.dsigmoid(zs[-l])
            #print(a)
            #print(b)
            
            delta = a * b
#            print("delta\n",delta)
#        
#            print(" ")
            #print("act\n",np.reshape(activations[-l-1], (-1,1)))

            delta_b[-l] = delta
            delta_w[-l] = delta * np.reshape(activations[-l-1], (-1,1))

            #print("delta_w\n",delta_w[-l])
        
        #print("\nFIM Backpropagation\n")

        #print(self.w[-2] - 0.5*delta_w[-2])
        
#        print("delta_w")
#        print(delta_w)
        return (delta_b,delta_w)
        
        
    def feedfoward(self, x):
        
        
        activation = x
        activations = [x] # list to store all the activations, layer by layer
       
        zs = [] # list to store all the z vectors, layer by layer
    
        for b, w in zip(self.b, self.w):

            z = np.dot(w, activation) + b.T[0]
            zs.append(z)
            
            activation = self.sigmoid(z)
            activations.append(activation)

        return activations, zs
    
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

#            print("w\n", w)
#            print("")
#            print("b\n", b)
#            print("x\n", x)
#            print("")
#            print("npdot", np.dot(w, x))
#            print("")
#            print("z", np.dot(w,x)+ b.T[0]) 
        
            x = self.sigmoid(np.dot(w, x) + b.T[0])
            #x = self.sigmoid(np.sum(np.dot(w, x), b))
#            print("\n\n")
            
        
        return x
#
    
    @staticmethod
    def sigmoid(x):
        
        # return np.heaviside(x, 1)
        
        return 1.0 / (1.0 + np.exp(-x));

    @staticmethod
    def dsigmoid(x):
        
        return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x));
        
        


if __name__ == "__main__":
    
    nn = NeuralNetwork([2,2,1], 0.24)
    
#    
#    dataset = ([[0,0],
#                        [0,1],
#                        [1,0],
#                        [1,1]])
#    
#    dataset2 = ([
#                         [0],
#                         [1],
#                         [1],
#                         [1]])
#    
#
    nn.trainModel(np.array([0.05, 0.1]), np.array([0.99]), 10)
    print("predição:", nn.predict([0.05,0.1]))
    #print("Vai toma no cu:", nn.feedfoward([0.05, 0.1]))

#    b,w= nn.backpropagation(np.array([0.05, 0.1]), np.array([0.01, 0.99]))
##    
#    print(nn.w[-1] - 0.5*np.array(w[-1]))
#    a,b = nn.feedfoward([0.05, 0.1])
#    print("Saida", a[-1])
##    #nn.backpropagation([0.05, 0.1], [0.01, 0.99])
#    print("Saida", nn.predict([1, 1]))
#
#    