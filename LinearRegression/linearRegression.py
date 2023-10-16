#多元线性回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#read data
raw_data = pd.read_csv("dataset/student/student-mat.csv",sep=';')

#print(raw_data.columns)
x = raw_data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences','G1','G2']]
y = raw_data['G3']

# data numeric
x['school'] = x['school'].map({'GP':0,"MS":1})
x['sex'] = x['sex'].map({"F":0,"M":1})
x['address'] = x['address'].map({"U":0,"R":1})
x['famsize'] = x['famsize'].map({"LE3":0,"GT3":1})
x['Pstatus'] = x['Pstatus'].map({"T":0,"A":1})
x = x.drop(columns="Mjob")
x = x.drop(columns="Fjob")
x = x.drop(columns="reason")
x['guardian'] = x['guardian'].map({"mother":0,"father":1,"other":3})
x['schoolsup'] = x['schoolsup'].map({"no":0,"yes":1})
x['famsup'] = x['famsup'].map({"no":0,"yes":1})
x['paid'] = x['paid'].map({"no":0,"yes":1})
x['activities'] = x['activities'].map({"no":0,"yes":1})
x['nursery'] = x['nursery'].map({"no":0,"yes":1})
x['higher'] = x['higher'].map({"no":0,"yes":1})
x['internet'] = x['internet'].map({"no":0,"yes":1})
x['romantic'] = x['romantic'].map({"no":0,"yes":1})

#split data to training and test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#buidl and training
model = LinearRegression()
model.fit(x_train,y_train)
print(model.coef_)

#predict
predictions = model.predict(x_test)
print(predictions,y_test)
plt.scatter(y_test,predictions)
plt.show()