# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:07:02 2024

@author: Priyanka
"""

"""
A National Zoopark in India is dealing with the problem of segregation
of the animals based on the different attributes they have. 
Build a KNN model to automatically classify the animals. 
Explain any inferences you draw in the documentation
Business Problem:-
Q.What is the business objective?
As zoos are visited by a large number of visitors, zoos are a point a potent
tool for education people about the close linkage between protection of natural
areas and maintaining the life supporting processes of nature.

Well-planned and appropriately designed zoos can sensitize 
visitors to the dangers of a hostile or
indifferent attitude towards nature. 

Q.Are there any constraints?
In India, many well designed zoos were set up in some of the States but for
the most part, zoos have not been able to meet the challenges imposed by the
changing scenario and still continue with the legacy of past i.e. displaying
animals to the animals nor educative and rewarding to the visitors.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
zoo=pd.read_csv("C:\Data Set\Zoo.csv")

#Exploratory Data Analysis (EDA):
zoo.dtypes
#All the inout features are of float type and output column i.e.type is integer type
zoo.columns
zoo.describe()
plt.hist(zoo.hair)

plt.hist(zoo.feathers)

#Since it is Multivariate, in the type column,only class numbers have been given
#let us change the type of class it belongs
zoo['type']=np.where(zoo['type']=='1','cat-1',zoo['type']) 
zoo['type']=np.where(zoo['type']=='2','cat-2',zoo['type'])
zoo['type']=np.where(zoo['type']=='3','cat-3',zoo['type'])
zoo['type']=np.where(zoo['type']=='4','cat-4',zoo['type'])
zoo['type']=np.where(zoo['type']=='5','cat-5',zoo['type'])
zoo['type']=np.where(zoo['type']=='6','cat-6',zoo['type'])
zoo['type']=np.where(zoo['type']=='7','cat-7',zoo['type'])
zoo.type

#All the columns having data in different scales ,hence need normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
zoo.shape
#The first column is a animal name let us ignore it,there for 1:16
zoo_norm=norm_func(zoo.iloc[:,1:17])
zoo_norm.describe()

# Training the model
#Before that,let us assign input and output columns
X=np.array(zoo_norm.iloc[:,:])
y=np.array(zoo['type'])

#let us split the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred
#Evaluate the accuracy and applicability of the model
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)
#Accuracy is 0.9523809523809523
pd.crosstab(pred,y_test,rownames = ['Actual'], colnames= ['Predictions'])

#Error on train data
pred_train=knn.predict(X_train)
accuracy_score(pred_train,y_train)
#0.0.9875
pd.crosstab(pred_train,y_train,rownames=['Actual'],colnames=['predicted'])

#Tunning of the model
#For selection of optimum value of k
acc=[]
#Let us run KNN on values 3,50 in step of 2 so that next value will be odd
for i in range(3,50,2):
    knn1=KNeighborsClassifier(n_neighbors=i)
    knn1.fit(X_train,y_train)
    train_acc=np.mean(knn1.predict(X_train)==y_train)
    test_acc=np.mean(knn1.predict(X_test)==y_test)
    acc.append([train_acc,test_acc])
#To plot the graph of accuracy of training and testing
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")    
plt.plot(np.arange(3,50,2),[i[1]for i in acc],"bo-")
#K=5 has got better accuracy where training accuracy and test accuracy are same
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred
#Evaluate the accuracy and applicability of the model
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)
#Accuracy is 0.9523809523809523
pd.crosstab(pred,y_test,rownames = ['Actual'], colnames= ['Predictions'])

#Error on train data
pred_train=knn.predict(X_train)
accuracy_score(pred_train,y_train)
#0.95
#still the model is over fit