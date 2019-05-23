import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
import matplotlib
# matplotlib.use('Qt5Agg')


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


# 打印聚类结果
# for i in range(16):
#     print('聚类{}\n中心：{}\n成员：{}\n'.format(i+1, centers[i], cluster_result[i]))

for i in range(len(labels)):
    plt.plot(list(df[centers[i]]))
    plt.title('cluster{}, center: {}'.format(i + 1, centers[i]))
    plt.savefig(r'img\cluster{}.jpg'.format(i + 1))
    plt.close()
