import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
data = iris.data # 输出一个150*4矩阵
# K-means聚类法原理手撕


def kmeans(sample_dataset, n_clusters, max_iter):
    num = sample_dataset.shape[0] # 样本数
    init_mean = sample_dataset[0:n_clusters] # 随机选取前n_clusters个向量作为初始均值向量
    repeated_sample = np.repeat(sample_dataset, n_clusters, axis=0)
    n = 0
    while (n!=max_iter):
        tiled_mean = np.tile(init_mean, (num,1)) # 对两个矩阵作处理以便于计算
        dist_all = np.sqrt(np.sum((repeated_sample - tiled_mean)**2, axis=1)) # 这里采用欧氏距离
        reshaped_dist = dist_all.reshape(num, n_clusters) # 重构矩阵，维度为num*n_clusters，每个数为样本距离初始均值向量的距离
        labels = reshaped_dist.argmin(axis=1) # 找距离最小的初始均值向量，提取其索引，形成类目标签
        stacked = np.hstack((sample_dataset, labels.reshape(num,1)))
        clusters = []
        for i in range(n_clusters):
            clusters.append(stacked[stacked[:,-1]==i,:])
            new_mean = np.mean(np.delete(clusters[i],-1,axis=1),axis=0)
            if not (new_mean == init_mean[i]).all():
                init_mean[i] = new_mean
        n = n + 1
    return labels


print(kmeans(data, 3, 1000))
print(iris.target)
