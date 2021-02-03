import pandas as pd
import numpy as np
import math

# 高斯分布模型的EM算法估计
class EMGaussianMixture:
    # 设初值为均匀分布
    def __init__(self, n_gaussian):
        self.n_gaussian = n_gaussian
        r = np.random.rand(n_gaussian)
        self.alpha = r / np.sum(r)
        self.mu = np.random.rand(n_gaussian) * 10
        self.sigma = np.random.rand(n_gaussian) * 10

    def gaussian(self, y):
        p = self.alpha * ((1 / (math.sqrt(2*math.pi) * self.sigma)) * \
               np.exp(-np.square(y - self.mu) / (2 * np.square(self.sigma))))
        if np.sum(p) == 0.0:
            gamma = np.zeros(len(p))
        else:
            gamma = p / np.sum(p)
        return gamma

    def fit(self, observed_value):
        y = np.array(observed_value).reshape(-1,)
        delta = 1000
        theta = np.concatenate([self.mu,self.sigma,self.alpha])
        while delta >= 0.1:
            gamma = np.hstack(np.frompyfunc(self.gaussian,1,1)(y)).reshape(-1,self.n_gaussian)
            sum = np.sum(gamma, axis=0)
            self.mu = np.sum(np.multiply(gamma, y.reshape(-1,1)), axis=0) / sum
            self.sigma = np.sqrt(np.sum(np.multiply(gamma, np.square(np.subtract(y.reshape(-1,1), self.mu))), axis=0) / sum)
            self.alpha = sum / len(y)
            theta_new = np.concatenate((self.mu,self.sigma,self.alpha))
            delta = np.sqrt(np.sum(np.square(theta - theta_new)))
            theta = theta_new
        return self.alpha, self.sigma, self.mu


if __name__ == '__main__':
    data = pd.DataFrame({"y":[-67,-48,6,8,14,16,23,24,28,29,41,49,56,60,75]})
    model = EMGaussianMixture(n_gaussian=2).fit(observed_value=data[["y"]])
    print("alpha:",end="")
    print(model[0])
    print("sigma:",end="")
    print(model[1])
    print("mu:",end="")
    print(model[2])
