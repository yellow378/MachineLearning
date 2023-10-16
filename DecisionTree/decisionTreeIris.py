from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score


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
clf = tree.DecisionTreeClassifier(criterion="gini")
clf.fit(x_train,y_train)
result = clf.score(x_test,y_test)
print(result)
# 输出dot格式的树状图，在这个里面我们将outfile设为None，也可以out_file='tree.dot'。然后可以转换成png格式：$ dot -Tpng tree.dot -o tree.png
dot_data = tree.export_graphviz(clf,
                                out_file = 'tree.dot',
                                feature_names = ['sepal length (cm)',
                                                'sepal width (cm)',
                                                'petal length (cm)',
                                                'petal width (cm)'],
                                class_names = ['setosa', 'versicolor', 'virginica'],
                                filled=True,
                                rounded=True
                               )

