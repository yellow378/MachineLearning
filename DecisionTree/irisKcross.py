from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression


#read data
rawdata = pd.read_csv("../LinearRegression/dataset/iris/iris.data",sep=',')
x = rawdata[['slength','swidth','plength','pwidth']]
y = rawdata[['class']]
#cy.loc[:,'class'] = y['class'].map({"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2})
# pd.set_option('display.max_columns', None)
# #显示所有行
# pd.set_option('display.max_rows', None)
# print(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#基尼系数
clf_gini = tree.DecisionTreeClassifier(criterion="gini")
clf_entropy  = tree.DecisionTreeClassifier(criterion="entropy")
lr = LogisticRegression(max_iter=10000)

scores_gini = cross_val_score(clf_gini,x,y,cv=5,scoring='accuracy')
scores_entropy = cross_val_score(clf_entropy,x,y,cv=5,scoring="accuracy")
yy = np.array(y).ravel()
scores_lr = cross_val_score(lr,x,yy,cv=5,scoring="accuracy")
print("gini:",scores_gini,scores_gini.mean())
print("entropy:",scores_entropy,scores_entropy.mean())
print("logistic regression:",scores_lr,scores_lr.mean())
