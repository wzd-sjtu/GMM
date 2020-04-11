

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#  下面构建鹫尾花数据集的分类
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

iris_dataset=load_iris()

X, X_test, y, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print(X)
X=X[:,0:3]
print(X)

#  这个一共有多少个不同的类别？
fig = plt.figure()
ax = Axes3D(fig)

gmm = GaussianMixture(n_components=3, random_state=42).fit(X)
labels = gmm.predict(X)
ax.scatter(X[:, 0], X[:, 1], X[:,2], c=labels, s=40, cmap='viridis')
plt.show()

print(labels)
print(y)

labels[np.where(labels==0)]=4
labels[np.where(labels==1)]=5
labels[np.where(labels==2)]=3
labels=labels-3

miss=0
total=0
for i,y in zip(labels,y):
    total+=1
    if i!=y:
        miss+=1

percent=1-miss/total
print(percent)