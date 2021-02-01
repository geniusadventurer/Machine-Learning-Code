import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import math

# 基本分类器，设置为x>v或x<v
def basic_classifier(x, v):
    if x < v:
        return 1
    else:
        return -1

# Adaboost提升算法
class AdaBoost:
    def __init__(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        self.N = len(x_train)
        self.w = np.ones(self.N) / self.N

    def clf(self, v):
        return np.frompyfunc(basic_classifier,2,1)(self.x, v)

    # 误差率计算
    def error_rate(self, v):
        prediction = self.clf(v)
        y = np.array(self.y)
        correctness = np.array([1 if prediction[i] != y[i] else 0 for i in range(self.N)])
        result = np.sum(np.multiply(self.w, correctness))
        return result

    # 模型拟合
    def fit(self, M):
        alpha = []
        bnds = ((self.x.min(),self.x.max()),) # 只有一组限制条件，也要做成(bnds,)的元组，否则读不出来
        v_lst = []
        for j in range(M):
            error = differential_evolution(func = self.error_rate,
                             bounds = bnds)
            e = error.fun
            v = error.x
            v_lst.append(error.x[0])
            alpha.append(0.5 * math.log((1-e)/e))
            exp = np.exp(list(-alpha[j] * np.array(self.y) * np.array(self.clf(v)))) # 太奇怪了，做成np.array就用不了np.exp，非要转成list，这个问题真的不知道是怎么回事
            self.w = np.divide( np.multiply(self.w, exp), np.sum(np.multiply(self.w, exp)) )
        self.v = v_lst
        self.alpha = alpha
        return self

if __name__ == '__main__':
    data = pd.DataFrame({"x":[0,1,2,3,4,5,6,7,8,9],
                         "y":[1,1,1,-1,-1,-1,1,1,1,-1]})
    model = AdaBoost(x_train=data["x"], y_train=data["y"]).fit(M=10)
    print("prediction:")
    print(model.alpha)
    print(model.v)
