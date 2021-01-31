import pandas as pd
import numpy as np
from scipy.optimize import minimize
import math

# 二元逻辑斯蒂回归
class LogisticRegression:
    # 对数似然函数
    def log_likelihood(self, w):
        result = 0
        for i in range(self.x.shape[0]-1):
            result = result + (np.sum(self.y[i] * (np.sum(np.multiply(w,self.x[i]))) - \
                                      np.log(1 + math.exp(np.sum(np.multiply(w,self.x[i]))))))
        return result

    # 模型拟合
    def fit(self, x, y):
        self.x = np.r_[np.array(x), [np.ones(x.shape[1])]] # 把1也放进x向量里
        self.y = np.array(y)
        w0 = np.zeros(x.shape[1])
        result = minimize(fun = self.log_likelihood,
                          x0 = w0,
                          method = 'SLSQP')
        self.w = result.x
        return self

    # 求概率
    def p(self, test):
        w = self.w
        result = []
        p = {}
        for i in range(test.shape[0]):
            p[0] = 1 / (1 + math.exp(np.sum(np.multiply(w, np.array(test.iloc[i])))))
            p[1] = 1 - p[0]
            for key,value in p.items():
                if value == max(p[0], p[1]):
                    result.append(key) # 把概率最大的值对应的y放入结果列表
        return result

if __name__ == '__main__':
    data = pd.DataFrame({"field1":[1,1.2,1.75,1.6,1.4,2,2.33,2.24,2.9,2.78,3.41,3.1,3.78,3.7,4.2],
                         "field2":[2.9,1.2,3.7,1.6,4.2,3.1,1.4,2,2.33,1.75,1,2.78,3.41,3.78,2.24],
                         "value":[0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]})
    test = pd.DataFrame({"field1":[1.7,2.3],
                         "field2":[3.89,1.04]})
    model = LogisticRegression().fit(x=data[["field1","field2"]], y=data["value"])
    print("prediction:")
    print(model.p(test=test))
