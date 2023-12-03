import joblib
from sklearn import datasets, model_selection
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

mnist = datasets.fetch_openml('mnist_784',version=1,cache=True,parser='auto')
data,target = mnist.data,mnist.target
train_X,test_X,train_y,test_y = model_selection.train_test_split(data,target,test_size=0.2)


print("loading...")
model = joblib.load('save/svm.pkl')


X,y = test_X.iloc[0],test_y.iloc[0]
X = X.values.reshape(1,-1)
image = np.array(X)
image = image.reshape(28,28)
y_hat = model.predict(X)
print('预测结果:',y_hat)
plt.imshow(image)
plt.show()