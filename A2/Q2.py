#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.data import loadlocal_mnist


# # Building MLP

# In[224]:


def relu(x):
    return np.maximum(0, x)

def reluder(x):
    return np.where(x<0, 0, 1)

def leaky(x):
    return np.maximum(0.01*x, x)

def leakyder(x):
    return np.where(x<0, 0.01, 1)

def sigmoid(x):
    return 1 / (1 +  np.exp(-x))

def sigder(x):
    return sigmoid(x) * (1 - sigmoid(x))

def linear(x):
    return x

def linder(x):
    return np.ones(len(x))

def tanh(x):
    return np.tanh(x)

def tander(x):
    return 1 - tanh(x)*tanh(x)

def softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    sfmx = numerator / denominator
    return sfmx

def softder(x):
    n = len(x)
    jacobian = np.zeros((n, n))
    soft = softmax(x)
    for i in range(n):
        for j in range(n):
            if i == j:
                jacobian[i][j] = soft[i] * (1-soft[i])
            else: 
                jacobian[i][j] = -soft[i]*soft[j]
    return jacobian


# In[270]:


class MyNeuralNetwork:
    def __init__(self):
        # defining default values for parameters
        pass

    def initialize(self, layers=4, lsize=[784, 256, 64, 10], acti='sigmoid', lr=0.1, weights='zero', bsize=32, epochs=5):
        self.n_layers = layers
        self.layer_sizes = lsize
        self.activation = acti
        self.learning_rate = lr
        self.weight_init = weights
        self.make_weights()
        self.batch_size = bsize
        self.num_epochs = epochs

    def loss(self, preds, labels):
        '''
        preds: (N, 10) ndrray
        labels: N labels
        '''
        N = preds.shape[0]
        ce = 0
        for i in range(N):
            ce += -np.log(preds[labels[i]] + 1e-9)
        ce /= N
        return ce
    
    def activator(self, x, acti):
        if(acti == 'relu'):
            return relu(x)
        if(acti == 'leaky'):
            return leaky(x)
        if(acti == 'sigmoid'):
            return sigmoid(x)
        if(acti == 'linear'):
            return linear(x)
        if(acti == 'tanh'):
            return tanh(x)
    
    def derivator(self, x, acti):
        if(acti == 'relu'):
            return reluder(x)
        if(acti == 'leaky'):
            return leakyder(x)
        if(acti == 'sigmoid'):
            return sigder(x)
        if(acti == 'linear'):
            return linder(x)
        if(acti == 'tanh'):
            return tander(x)
    
    def make_weights(self):
        if(self.weight_init == 'zero'):
            self.wzero()
        elif(self.weight_init == 'random'):
            self.wrandom()
        else:
            self.wnormal()
    
    def wzero(self):
        self.weights = []
        self.biases = []
        for i in range(1, self.n_layers):
            self.weights.append(np.zeros((self.layer_sizes[i], self.layer_sizes[i-1])))
            self.biases.append(np.zeros(self.layer_sizes[i]))
    
    def wrandom(self):
        self.weights = []
        self.biases = []
        for i in range(1, self.n_layers):
            self.weights.append(np.random.randn((self.layer_sizes[i], self.layer_sizes[i-1])) * 0.01)
            self.biases.append(np.zeros(self.layer_sizes[i]))
    
    def wnormal(self):
        self.weights = []
        self.biases = []
        for i in range(1, self.n_layers):
            self.weights.append(np.random.normal((self.layer_sizes[i], self.layer_sizes[i-1])) * 0.01)
            self.biases.append(np.zeros(self.layer_sizes[i]))
    
    def onehot(self, y):
        y2 = np.zeros((len(y), 10))
        for i, label in enumerate(y):
            y2[i][label] = 1
        return y2
    
    def forward_pass(self, x):
        # takes one sample as input x
        # returns probaabilities for that sample
        al = []
        zl = []
        for i in range(self.n_layers-1):
            z = np.dot(self.weights[i], x) + self.biases[i]
            a = self.activator(z, self.activation)
            al.append(a)
            zl.append(z)
            x = a
        probs = softmax(x)
        return [al, zl, probs]
    
    def backprop(self, x, y, params):
        al = params[0]
        zl = params[1]
        probs = params[2]
        wgrads = []
        bgrads = []
        # one hot encoding for y
        y2 = onehot(y)
        for i in range(self.n_layers-2, -1, -1):
            if(i == (self.n_layers-2)):
                dz = (probs - y2) / x.shape[0]
                dw = np.dot(al[i].T, dz)
            else:
                dz = np.dot(self.weights[i+1].T, probs-y2) / x.shape[0]
                if(i==0):
                    dw = np.dot(x, dz.T)
                else:
                    dw = np.dot(zl[i-1], dz.T)
            db = np.sum(dz) / x.shape[0]
            wgrads.append(dw)
            bgrads.append(db)
        wgrads = wgrads[::-1]
        bgrads = bgrads[::-1]
        return [wgrads, bgrads]
    
    def update_weights(self, grads):
        wgrads = grads[0]
        bgrads = grads[1]
        for i in range(self.n_layers-1):
            self.weights[i] = self.weights[i] - self.learning_rate * wgrads[i]
            self.biases[i] = self.biases[i] - self.learning_rate * bgrads[i]
    
    def fit(self, x, y, xv, yv):
        n = x.shape[0]
        m = x.shape[1]
        batches = n // self.batch_size
        
        self.train_loss = []
        self.val_loss = []
        
        for epoch in range(self.num_epochs):
            for i in range(0, n, self.batch_size):
                x0 = x[i:i+self.batch_size]
                y0 = y[i:i+self.batch_size]
                params = self.forward_pass(x0)
                grads = self.backprop(x0, y0, params)
                self.update_weights(grads)
            
            probs = self.forward_pass(x)[2]
            self.train_loss.append(self.loss(probs.T))
            probs = self.forward_pass(xv)[2]
            self.val_loss.append(self.loss(probs.T))
    
    def predict_proba(self, x):
        params = self.forward_pass(x)
        return params[2].T
    
    def predict(self, x):
        params = self.forward_pass(x)
        probs = params[2].T
        pred = np.zeros(len(x))
        for i in range(len(x)):
            pred[i] = probs[i].argmax(axis=0)
        return pred
    
    def score(self, x, y):
        pred = self.predict(x)
        return np.sum(pred==y)


# In[ ]:





# In[ ]:





# # Testing the model

# #### Only the training images are utilised and split into 7:2:1 train:test:val split

# In[2]:


X, Y = loadlocal_mnist(images_path='data/train-images.idx3-ubyte', labels_path='data/train-labels.idx1-ubyte')


# In[3]:


print('Images: ' + str(X.shape))
print('Labels: ' + str(Y.shape))


# In[4]:


x_train = X[:42000]
y_train = Y[:42000]
x_test = X[42000:54000]
y_test = Y[42000:54000]
x_val = X[54000:]
y_val = Y[54000:]


# In[5]:


print('x_train: ' + str(x_train.shape))
print('y_train: ' + str(y_train.shape))
print('x_test:  '  + str(x_test.shape))
print('y_test:  '  + str(y_test.shape))
print('x_val:  '  + str(x_val.shape))
print('y_val:  '  + str(y_val.shape))


# In[6]:


#separating training data into classes and storing indexes
ci = []
for i in range(10):
    ci.append(np.argwhere(y_train == i))


# #### Printing 5 sample images for each class

# In[7]:


for i in range(10):
    plt.figure(figsize=(10, 50))
    count = 0
    for j in range(5):
        plt.subplot(1, 5, j+1)
        plt.imshow(x_train[ci[i][j]].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()


# ## Testing the model

# In[260]:


funcs = ['relu', 'leaky', 'sigmoid', 'linear', 'tanh']


# In[271]:


model_weights = {'relu': {'weights': [], 'biases': []}, 'leaky': {'weights': [], 'biases': []},
                'sigmoid': {'weights': [], 'biases': []}, 'linear': {'weights': [], 'biases': []}, 
                'tanh': {'weights': [], 'biases': []}}

model_loss = {'relu': {'train': [], 'valid': []}, 'leaky': {'train': [], 'valid': []},
                'sigmoid': {'train': [], 'valid': []}, 'linear': {'train': [], 'valid': []}, 
                'tanh': {'train': [], 'valid': []}}

for func in funcs:
    model = MyNeuralNetwork()
    model.initialize(layers=6, lsize=[784, 256, 128, 64, 32, 10], acti='func', lr=0.08, weights='normal', bsize=1000, epochs=150)
    model.fit(x_train, y_train, x_val, y_val)
    model_weights[func]['weights'] = model.weights
    model_weights[func]['biases'] = model.biases
    model_loss[func]['train'] = model.train_loss
    model_loss[func]['valid'] = model.val_loss


# In[ ]:




