#multiclass ovr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
#read data
raw_data = pd.read_csv("dataset/winequality/winequality-white.csv",sep=';')
x = raw_data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
y = raw_data['quality']

#split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#build and train
model = OneVsRestClassifier(SVC())
model.fit(x_train,y_train)

#predict
prediction = model.predict(x_test)

pl = prediction.tolist()
yl = y_test.tolist()
yes = 0
sum = len(pl)
for i in range(0,sum):
    if pl[i] == yl[i]:
        yes = yes + 1
print("{}/{}".format(yes,sum))
#result 
result = model.score(x_test,y_test)
print(result)