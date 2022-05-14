#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# In[2]:


xTrain,yTrain=loadlocal_mnist(images_path='./mnist/train-images.idx3-ubyte', labels_path='./mnist/train-labels.idx1-ubyte')
xTest,yTest = loadlocal_mnist(images_path='./mnist/t10k-images.idx3-ubyte', labels_path='./mnist/t10k-labels.idx1-ubyte')


# In[3]:


print(xTrain.shape,yTrain.shape)


# In[4]:


#class for FDA
class FDA:
    W = None
    eigen_values = None
    Sw, Sb = None, None

    def __init__(self):
        W = None
        
    def fit(self, X, Y):
        Sw,Sb = self.get_scatter_within_between(X, Y)
        self.Sw = Sw
        self.Sb = Sb
        Sw_inv_Sb = np.matmul(np.linalg.inv(Sw), Sb)
        eig_values, eig_vectors = np.linalg.eigh(Sw_inv_Sb)
        idx = eig_values.argsort()[::-1]   
        eig_vectors = eig_vectors[:,idx]
        eig_values = eig_values[idx]
        rank = np.linalg.matrix_rank(Sb)
        eig_vectors = eig_vectors[:,:rank]
        eig_values  = eig_values[:rank]
        self.eigen_values = eig_values
        self.W = eig_vectors

    def get_scatter_within_between(self, X, Y):
        Si = []
        unique_Y = np.unique(Y)
        for label in unique_Y:
            indexes = np.where(Y == label)[0]
            selected_X = X[indexes]
            selected_X_trans = selected_X.T
            scatter_matrix = np.cov(selected_X_trans, ddof=0) * selected_X_trans.shape[1]
            Si.append(scatter_matrix)
        Sw = np.zeros(Si[0].shape)
        for scatter_matrix in Si:
            Sw += scatter_matrix
            
        X_tran = X.T
        St = np.cov(X_tran, ddof=0) * X_tran.shape[1]
        Sb = St - Sw
        
        return Sw,Sb


# In[8]:


scaler = StandardScaler()
scaler.fit(xTrain)
pca = PCA(n_components=15)

xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)

pca.fit(xTrain)
xTrain = pca.transform(xTrain)
xTest = pca.transform(xTest)

fda = FDA()
fda.fit(xTrain,yTrain)

W = fda.W
xTrainProjected = np.matmul(W.T,xTrain.T).T
xTestProjected = np.matmul(W.T,xTest.T).T

clf = LinearDiscriminantAnalysis()
clf.fit(xTrainProjected,yTrain)

print("Accuracy we get is - ", clf.score(xTestProjected, yTest)*100,"%")


# In[11]:


yPredict = clf.predict(xTestProjected)
matrix = confusion_matrix(yTest, yPredict)
accuracyClass = matrix.diagonal()/matrix.sum(axis=1)
for i in range(10):
    print("Accuracy for class "+str(i)+" = ",accuracyClass[i]*100,"%")


# In[ ]:




