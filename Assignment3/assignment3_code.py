#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# In[4]:


def loadData():
    xTrain=[]
    yTrain=[]
    xTest=[]
    yTest=[]
    
    for i in ['./cifar-10-batches-py/data_batch_1','./cifar-10-batches-py/data_batch_2','./cifar-10-batches-py/data_batch_3','./cifar-10-batches-py/data_batch_4','./cifar-10-batches-py/data_batch_5']:
        file = open(i,'rb')
        data = pickle.load(file,encoding='bytes')
        xTrain += list(data[b'data'])
        yTrain += data[b'labels']
    
    file = open('./cifar-10-batches-py/test_batch','rb')
    data = pickle.load(file,encoding='bytes')
    xTest += list(data[b'data'])
    yTest += data[b'labels']
    
    files2 = open('./cifar-10-batches-py/batches.meta','rb')
    data = pickle.load(files2,encoding='bytes')
    labels = [i.decode('utf-8') for i in data[b'label_names']]
    
    return np.array(xTrain),np.array(yTrain),np.array(xTest),np.array(yTest),np.array(labels)
        


# In[5]:


xTrain,yTrain,xTest,yTest,labels = loadData()


# In[6]:


print(xTrain.shape,yTrain.shape,xTest.shape,yTest.shape,labels)


# In[19]:


#visualising as images
images = xTrain.reshape(len(xTrain),3,32,32).transpose(0,2,3,1)
for i in range(10):
    imageInd = np.where(yTrain==i)[0]
    fiveImageInd = imageInd[:5]
    print("Printing images of ",labels[yTrain[fiveImageInd[0]]])
    for j in fiveImageInd:
        plt.imshow(images[j])
        plt.show()


# In[20]:


#LDA
clf = LinearDiscriminantAnalysis()
clf.fit(xTrain,yTrain)
print("Accuracy on testing data - ",clf.score(xTest,yTest))


# In[22]:


yPredict = clf.predict(xTest)
matrix = confusion_matrix(yTest, yPredict)
accuracyClass = matrix.diagonal()/matrix.sum(axis=1)
for i in range(len(labels)):
    print("Accuracy for class "+labels[i]+" = ",accuracyClass[i])


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[10]:


xTrain,yTrain=loadlocal_mnist(images_path='./mnist/train-images.idx3-ubyte', labels_path='./mnist/train-labels.idx1-ubyte')
xTest,yTest = loadlocal_mnist(images_path='./mnist/t10k-images.idx3-ubyte', labels_path='./mnist/t10k-labels.idx1-ubyte')


# In[17]:


ncomponents = [3,8,15]
allAccuracy = []

for n in ncomponents:
    scaler = StandardScaler()
    scaler.fit_transform(xTrain)
    pca = PCA(n_components = n)
    transform = pca.fit_transform(xTrain)
    lda = LinearDiscriminantAnalysis()
    lda.fit(transform,yTrain)
    transformTest = pca.transform(xTest)
    yPredict = lda.predict(transformTest)
    allAccuracy.append(100*np.sum(yPredict == yTest)/len(yTest))

plt.figure("Plotting accuracies")
plt.xlabel("n_components")
plt.ylabel("Accuracy (%)")
plt.plot(ncomponents,allAccuracy)
plt.title("Accuracy vs n_components")
plt.show()


# In[ ]:
#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[2]:


xTrain,yTrain=loadlocal_mnist(images_path='./fmnist/train-images-idx3-ubyte', labels_path='./fmnist/train-labels-idx1-ubyte')
xTest,yTest = loadlocal_mnist(images_path='./fmnist/t10k-images-idx3-ubyte', labels_path='./fmnist/t10k-labels-idx1-ubyte')


# In[3]:


print(xTrain.shape,yTrain.shape)


# In[11]:


#class for FDA
#class for FDA
class FDA:
    W = None
    eigenValues = None
    Sw, Sb = None, None

    def __init__(self):
        W = None
        eigenValues = None
        Sw, Sb = None, None
        
    def fit(self, x, y):
        Sw,Sb = self.getScatters(x, y)
        self.Sw = Sw
        self.Sb = Sb
        Sw_inv_Sb = np.matmul(np.linalg.inv(Sw), Sb)
        eigValues, eigVectors = np.linalg.eigh(Sw_inv_Sb)
        idx = eigValues.argsort()[::-1]   
        eigVectors = eigVectors[:,idx]
        eigValues = eigValues[idx]
        rank = np.linalg.matrix_rank(Sb)
        eigVectors = eigVectors[:,:rank]
        eigValues  = eigValues[:rank]
        self.eigenValues = eigValues
        self.W = eigVectors

    def getScatters(self, x, y):
        Si = []
        uniqueY = np.unique(y)
        for label in uniqueY:
            indexes = np.where(y == label)[0]
            selectedX = x[indexes]
            selectedXTrans = selectedX.T
            scatterMatrix = np.cov(selectedXTrans, ddof=0) * selectedXTrans.shape[1]
            Si.append(scatterMatrix)
        Sw = np.zeros(Si[0].shape)
        for scatterMatrix in Si:
            Sw += scatterMatrix
            
        xTrans = x.T
        St = np.cov(xTrans, ddof=0) * xTrans.shape[1]
        Sb = St - Sw
        
        return Sw,Sb


# In[14]:


fda = FDA()
fda.fit(xTrain,yTrain)

W = fda.W

xTrainProjected = np.matmul(W.T,xTrain.T).T
xTestProjected = np.matmul(W.T,xTest.T).T

clf = LinearDiscriminantAnalysis()
clf.fit(xTrainProjected,yTrain)

print("Accuracy by own fda = ",clf.score(xTestProjected,yTest)*100,"%")


# In[ ]:
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
#class for FDA
class FDA:
    W = None
    eigenValues = None
    Sw, Sb = None, None

    def __init__(self):
        W = None
        eigenValues = None
        Sw, Sb = None, None
        
    def fit(self, x, y):
        Sw,Sb = self.getScatters(x, y)
        self.Sw = Sw
        self.Sb = Sb
        Sw_inv_Sb = np.matmul(np.linalg.inv(Sw), Sb)
        eigValues, eigVectors = np.linalg.eigh(Sw_inv_Sb)
        idx = eigValues.argsort()[::-1]   
        eigVectors = eigVectors[:,idx]
        eigValues = eigValues[idx]
        rank = np.linalg.matrix_rank(Sb)
        eigVectors = eigVectors[:,:rank]
        eigValues  = eigValues[:rank]
        self.eigenValues = eigValues
        self.W = eigVectors

    def getScatters(self, x, y):
        Si = []
        uniqueY = np.unique(y)
        for label in uniqueY:
            indexes = np.where(y == label)[0]
            selectedX = x[indexes]
            selectedXTrans = selectedX.T
            scatterMatrix = np.cov(selectedXTrans, ddof=0) * selectedXTrans.shape[1]
            Si.append(scatterMatrix)
        Sw = np.zeros(Si[0].shape)
        for scatterMatrix in Si:
            Sw += scatterMatrix
            
        xTrans = x.T
        St = np.cov(xTrans, ddof=0) * xTrans.shape[1]
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












