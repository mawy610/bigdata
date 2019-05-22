import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
import matplotlib
matplotlib.use('Qt5Agg')


df = pd.read_csv(r'data\all2.csv', index_col=0)
df.columns = list(map(lambda x: x[:8], df.columns))

X = []
for i in list(df):
    X.append(list(df[i]))

ap = AffinityPropagation().fit(X)
cluster_centers_indices = ap.cluster_centers_indices_
labels = ap.labels_
n_clusters_ = len(cluster_centers_indices)

centers = [list(df)[i] for i in cluster_centers_indices]
cluster_result = [[] for i in range(n_clusters_)]
for i in range(len(X)):
    cluster_result[labels[i]].append(list(df)[i])

plt.close('all')    # 关闭所有的图形
plt.figure(1)    # 产生一个新的图形
plt.clf()    # 清空当前的图形

plt.plot(df[list(df)[1]])
plt.show()