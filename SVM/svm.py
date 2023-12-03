from sklearn import datasets, model_selection
from sklearn import svm

mnist = datasets.fetch_openml('mnist_784',version=1,cache=True,parser='auto')
data,target = mnist.data,mnist.target
train_X,test_X,train_y,test_y = model_selection.train_test_split(data,target,test_size=0.2)


model = svm.SVC(gamma='scale',C=1.0,kernel='rbf',decision_function_shape='ovr')


print("training....")
model.fit(train_X,train_y)


print("accuray:")
result = model.score(test_X,test_y)
print(result) 


import joblib
print("saving...")
joblib.dump(model,'save/svm.pkl')