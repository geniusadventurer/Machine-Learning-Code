import numpy as np
import pandas as pd

class KMeans:
    def fit(self, sample_dataset, n_clusters, max_iter):
        sample_dataset = np.array(sample_dataset)
        num = sample_dataset.shape[0] # 样本数
        rand = np.random.choice(a=num, size=n_clusters, replace=False)
        init_mean = sample_dataset[rand] # 随机选取n_clusters个向量作为初始均值向量
        tiled_mean = np.tile(init_mean, (num,1)) # 对两个矩阵作处理以便于计算
        repeated_sample = np.repeat(sample_dataset, n_clusters, axis=0)
        n = 0
        while n != max_iter:
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

if __name__ == '__main__':
    data = pd.DataFrame({"x1": [4,8,6,2,9,7,1,5,6,9],
                         "x2": [7,0,5,1,3,4,4,2,8,3]})
    model = KMeans()
    label = model.fit(sample_dataset=data, n_clusters=4, max_iter=1000)
    print(label)
