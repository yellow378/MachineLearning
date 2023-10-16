from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from ucimlrepo import fetch_ucirepo 
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
from matplotlib.font_manager import FontProperties
# fetch dataset 
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
X = wine.data.features 
y = wine.data.targets 

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#基尼系数
clf_gini = tree.DecisionTreeClassifier(criterion="gini")
scores_gini = cross_val_score(clf_gini,X,y,cv=5,scoring='accuracy')
#信息熵
clf_entropy  = tree.DecisionTreeClassifier(criterion="entropy")
scores_entropy = cross_val_score(clf_entropy,X,y,cv=5,scoring="accuracy")

#K-fold 交叉对比
print("K折交叉精度对比:")
print("gini:",scores_gini, "平均值:",scores_gini.mean())
print("entropy:",scores_entropy,"平均值:",scores_entropy.mean())
print("\n")
gini_scores = []
entorpy_scores = []
gini_scores.append(round(scores_gini.mean(),2))
entorpy_scores.append(round(scores_entropy.mean(),2))

#训练预测
clf_entropy.fit(x_train,y_train)
clf_gini.fit(x_train,y_train)
y_entropy_pred = clf_entropy.predict(x_test)
y_gini_pred = clf_gini.predict(x_test)


#查准率（精确率）
a = precision_score(y_test,y_gini_pred,average='macro')
b = precision_score(y_test,y_entropy_pred,average='macro')
print("查准率对比:")
print("gini:",a)
print("entropy:",b)
print("\n")
gini_scores.append(round(a,2))
entorpy_scores.append(round(b,2))
#查全率
a = recall_score(y_test,y_gini_pred,average='macro')
b = recall_score(y_test,y_entropy_pred,average='macro')
print("查全率对比:")
print("gini:",a)
print("entropy:",b)
print("\n")
gini_scores.append(round(a,2))
entorpy_scores.append(round(b,2))
#F1
a = f1_score(y_test,y_gini_pred,average='macro')
b = f1_score(y_test,y_entropy_pred,average='macro')
print("F1对比:")
print("gini:",a)
print("entropy:",b)
print("\n")
gini_scores.append(round(a,2))
entorpy_scores.append(round(b,2))

my_font = FontProperties(fname=r"c:\windows\fonts\SimHei.ttf",size=12)
plt.figure(figsize=(13,4))
labels = ['准确率','查准率','查全率','f1']
plt.subplot(131)
x = np.arange(len(labels))  # x轴刻度标签位置
width = 0.25  # 柱子的宽度
# 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# x - width/2，x + width/2即每组数据在x轴上的位置
bar1 = plt.bar(x - width/2, gini_scores, width, label='gini')
bar2 = plt.bar(x + width/2, entorpy_scores, width, label='entropy')
plt.bar_label(bar1,label_type='edge')
plt.bar_label(bar2,label_type='edge')
plt.ylabel('分数',fontproperties=my_font)
plt.title('对比',fontproperties=my_font)
# x轴刻度标签位置不进行计算
plt.xticks(x, labels=labels,fontproperties=my_font)
plt.legend()
plt.tight_layout()
plt.show()