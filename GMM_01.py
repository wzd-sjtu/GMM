

import matplotlib.pyplot as plt

'''
Seaborn是基于matplotlib的图形可视化python包。它提供了一种高度交互式界面，
便于用户能够做出各种有吸引力的统计图表。

Seaborn是在matplotlib的基础上进行了更高级的API封装，
从而使得作图更加容易，在大多数情况下使用seaborn能做出很具有吸引力的图，
而使用matplotlib就能制作具有更多特色的图。
应该把Seaborn视为matplotlib的补充，而不是替代物。
同时它能高度兼容numpy与pandas数据结构以及scipy与statsmodels等统计模式。

作者：留心的话没有小事
链接：https://www.jianshu.com/p/94931255aede

'''

#  总的来说  还是调库比较方便的
import seaborn as sns
sns.set()
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
#  这里的生成数据不妨认为是正确的
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting
plt.scatter(X[:,0],X[:,1],s=40)
plt.show()
import numpy as np

#  random_state是默认随机数种子  这个需要自己进行设定
gmm = GaussianMixture(n_components=4 , random_state=42).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.show()

#  这里的距离太过接近导致出现了问题
X[:,0]=X[:,0]/20
gmm = GaussianMixture(n_components=4 , random_state=42).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.show()

rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))
gmm = GaussianMixture(n_components=4 , random_state=42).fit(X_stretched)
labels = gmm.predict(X_stretched)
plt.scatter(X_stretched[:, 0], X_stretched[:, 1], c=labels, s=40, cmap='viridis')
plt.show()