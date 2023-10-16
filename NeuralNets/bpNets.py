from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
x = iris.data.features 
y = iris.data.targets 
y = np.array(y).ravel()

#split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#build and train  
model = MLPClassifier(hidden_layer_sizes=(10,10),max_iter=10000)
model.fit(x_train,y_train)
#print(model.coefs_)
#print(model.intercepts_)
#result
result = model.score(x_test,y_test)
print(result)
