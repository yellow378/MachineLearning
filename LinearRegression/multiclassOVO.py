#multiclass ovo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo 

#read data
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 
  
# data (as pandas dataframes) 
X = optical_recognition_of_handwritten_digits.data.features 
y = optical_recognition_of_handwritten_digits.data.targets 
y = np.array(y)
#y = pd.get_dummies(y,columns=['class'])
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#y_train = y_train.to_numpy().reshape(1681,1)
#build and train
model = OneVsOneClassifier(SVC())
model.fit(x_train,y_train.ravel())

#predict
prediction = model.predict(x_test)

pl = prediction.tolist()
yl = y_test.ravel().tolist()
yes = 0
sum = len(pl)
for i in range(0,sum):
    if pl[i] == yl[i]:
        yes = yes + 1
print("{}/{}".format(yes,sum))

result = model.score(x_test,y_test)
print(result)