from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 

#read data
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 
  
# data (as pandas dataframes) 
X = optical_recognition_of_handwritten_digits.data.features 
y = optical_recognition_of_handwritten_digits.data.targets 
y = np.array(y)
#y = pd.get_dummies(y,columns=['class'])
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#信息熵
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf.fit(x_train,y_train)
result = clf.score(x_test,y_test)
print(result)
# 输出dot格式的树状图，在这个里面我们将outfile设为None，也可以out_file='tree.dot'。然后可以转换成png格式：$ dot -Tpng tree.dot -o tree.png
dot_data = tree.export_graphviz(clf,
                                out_file = 'tree.dot',
                                feature_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                                          '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
                                          '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
                                          '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
                                          '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61',
                                          '62', '63','64'],
                                class_names = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                filled=True,
                                rounded=True
                               )


#基尼系数
clf = tree.DecisionTreeClassifier(criterion="gini")
clf.fit(x_train,y_train)
result = clf.score(x_test,y_test)
print(result)

#对数损失函数
clf = tree.DecisionTreeClassifier(criterion="log_loss")
clf.fit(x_train,y_train)
result = clf.score(x_test,y_test)
print(result)