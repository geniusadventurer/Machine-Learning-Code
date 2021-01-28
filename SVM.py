import numpy as np
import pandas as pd
from scipy.optimize import minimize

def sign(v):
    if v >= 0:
        return 1
    else:
        return -1

class LinearSVM:
    # 目标函数
    def min_func(self, w):
        result = 0.5*(np.sum(np.square(w[:-1])))
        return result

    # 条件函数
    def cons_func(self, w):
        goal = np.sum(np.multiply(w[:-1], self.x), axis=1) + w[-1]
        result = np.multiply(self.y, goal) - 1
        return result

    # 模型拟合
    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        cons = [{'type':'ineq', 'fun':self.cons_func} for i in range(x.shape[0])]
        w0 = np.zeros(x.shape[1]+1)
        result = minimize(fun = self.min_func,
                          x0 = w0,
                          constraints = cons,
                          method = 'SLSQP')
        self.w = result.x[:-1]
        self.b = result.x[-1]
        goal = np.sum(np.multiply(self.w, x), axis=1) + self.b
        prediction = np.frompyfunc(sign, 1, 1)(goal)
        self.prediction = prediction
        return self

if __name__ == '__main__':
    data = pd.DataFrame({"field1":[3,4,1],
                         "field2":[3,3,1],
                         "value":[1,1,-1]})
    X = data[["field1","field2"]]
    Y = data["value"]
    model = LinearSVM()
    params = model.fit(x=X, y=Y)
    print("Model:")
    print("w=", end="")
    print(model.w)
    print("b=", end="")
    print(model.b)
    print("prediction_results:")
    print(model.prediction)
    print("\n", end="")
