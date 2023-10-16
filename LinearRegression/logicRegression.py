#logic regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#read data
raw_data = pd.read_csv("dataset/Algerian_forest_fires_dataset_UPDATE.csv",sep=',')
raw_data['Classes'] = raw_data['Classes'].map({"not fire":0,"fire":1})
x = raw_data[['day', 'month', 'year', 'Temperature', ' RH', ' Ws', 'Rain ', 'FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']]
y = raw_data['Classes']

# print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

model = LogisticRegression()
model.fit(x_train,y_train)
print(model.coef_)

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