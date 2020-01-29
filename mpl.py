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

        self.learning_rate = learning_rate
        

        
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

            for i in range(self.num_layers-1):
#                print("selfb", self.b[-i])
                print("Layers ", i)
                #print("b", b[-i])
                                
                if(len(self.predict([0.1, 0.1])) != self.layers[-1]):
                    raise Exception("ante")

                print("Derivs parciais")
                for b in w:
                    print(b)
                print("-----")
                
                
                print(self.w)
                print(w[i])
                self.w[i] = self.w[i] - self.learning_rate*(w[i])
                #self.b[-i] = self.b[-i] - self.learning_rate*(b[-i])
                
                
                
                print("Pesos")
                for x in self.w:
                    print(x)
                
                if (len(self.predict([0.1, 0.1])) != self.layers[-1]):
                    raise Exception("Depois")


        
            
            
  
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
        assert len(activations[-1]) == self.layers[-1]
        
        print(activations)
        print(zs)
        # cria um vetor que recebera cada delta_b e delta_w associado ao peso e vies correspondente
        
        delta_b = [np.zeros(b.shape) for b in self.b]
        delta_w = [np.zeros(w.shape) for w in self.w]
        
        print("\nInit Backpropagation\n")
        
        print("Camada de saída\n")
       
        
        dCdA = activations[-1] - y
        assert activations[-1].shape == y.shape and dCdA.shape == activations[-1].shape
        
        delta = dCdA * self.dsigmoid(zs[-1])
        assert len(delta) == self.layers[-1] and delta.shape == dCdA.shape
        
    

        print(delta)
        print(activations[-2])
        
        teste = []
        for i in activations[-2].reshape(-1,1):
            teste.append([i*delta])
            
#        print("Teste")
#        print(*teste, sep='\n')
        
        
#        for i,j in zip(np.transpose(teste), self.w[-1]):
#            print("w",j-i[0])
#        print("Pesos")
      
        #delta_w[-1] = np.dot(delta, np.transpose(activations[-2]))

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.dsigmoid(z)
            delta = np.dot(self.w[-l+1].T, delta) * sp
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, np.transpose(activations[-l-1]))
            
        return (delta_b, delta_w)

        
        
        
        

        print("\nCamandas ocultas\n")
        
        #for l in range(2, self.num_layers):
            
          
        return (delta_b,delta_w)
        
    
    
    
    def backprop(self, x, y):
        
        
        act, z = self.feedfoward(x)
        
        dB = [np.zeros(b.shape) for b in self.b]
        dW = [np.zeros(w.shape) for w in self.w]
        
        g = act[-1] - y
        assert g.shape == act[-1].shape
        
        for l in range(1, self.num_layers):
            
            g = g * self.dsigmoid(z[-l])
            assert g.shape == act[-1].shape            
            
            try:
               
                aux = [ i*act[-l-1] for i in g]
                dW[-l] = aux
            
            except ValueError:
                print("mds n guento mais")
                dW[-l] = g * act[-l-1].T
                
            dB[-l] = g

            g = np.dot(self.w[-l].T, g)
            
            
        return dW,dB
        
        
        
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
    def sigmoid(x):
                
        return 1.0 / (1.0 + np.exp(-x));

    @staticmethod
    def dsigmoid(x):
        
        return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x));
        
        


if __name__ == "__main__":
    
    np.random.seed(0)
    nn = NeuralNetwork([10,7], 0.01)
    
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
    #nn.trainModel(np.array([0.05, 0.1]), np.array([0.01, 2, 1]), 10000)
    
    for i in range(10000):
       # dw,db= nn.backprop(np.array([0.3, 1.1, 1, 2, 3]), np.array([0.1, 0.8, 0.4, 0.1,0.1,0.1, 0.3]))
        dw,db= nn.backprop(np.arange(1,11), np.array([0.1, 0.8, 0.4, 0.1,0.1,0.1, 0.3]))

        dw = dw[0]        
        
        print("cu",nn.w[0], sep = "\n" )
        print(dw, sep = "\n")
        
        nn.w[0] = nn.w[0] - nn.learning_rate*np.array(dw)
        nn.b = nn.b - nn.learning_rate*np.array(db)
        
    print("predição:", nn.predict(np.arange(1,11)))
#    #print("Vai toma no cu:", nn.feedfoward([0.05, 0.1]))
#
#    b,w= nn.backpropagation(np.array([0.05, 0.1]), np.array([0.01, 0.99, 1]))
##    
#    print(nn.w[-1] - 0.5*np.array(w[-1]))
#    a,b = nn.feedfoward([0.05, 0.1])
#    print("Saida", a[-1])
##    #nn.backpropagation([0.05, 0.1], [0.01, 0.99])
#    print("Saida", nn.predict([1, 1]))
#
#    