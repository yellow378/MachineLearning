from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
x = wine_quality.data.features 
y = wine_quality.data.targets 
y = np.array(y).ravel()

#split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#build and train
model = MLPRegressor(hidden_layer_sizes=(3,3,3),max_iter=10000)
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
