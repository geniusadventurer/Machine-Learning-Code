import numpy as np
import pandas as pd
import random

# 目标函数
def sign(v):
    if v >= 0:
        return 1
    else:
        return -1

class Perceptron:
    # 模型拟合
    def fit(self, x, y, w0, b0, learning_rate):
        x = np.array(x)
        y = np.array(y)
        w = np.array(w0)
        b = np.array(b0)
        goal = np.sum(np.multiply(w, x), axis=1) + b
        result = np.multiply(y, goal)
        correctness = (result > 0)
        while not correctness.all():
            where = np.where(result <= 0)
            i = random.choice(where[0].tolist())
            w = w - learning_rate * (-x[i]*y[i])
            b = b - learning_rate * (-y[i])
            goal = np.sum(np.multiply(w, x), axis=1) + b
            result = np.multiply(y, goal)
            correctness = (result > 0)
        prediction = np.frompyfunc(sign, 1, 1)(goal)
        self.w = w
        self.b = b
        self.prediction = prediction
        return self

class PerceptronDual:
    # Gram矩阵
    def gram(self, x):
        x = np.array(x)
        result = np.dot(x, x.T)
        return result

    # 模型拟合
    def fit(self, x, y, alpha0, b0, learning_rate):
        x = np.array(x)
        y = np.array(y)
        alpha = np.array(alpha0)
        b = np.array(b0)
        goal = np.multiply(alpha*y, self.gram(x)).sum(axis=1) + b
        result = np.multiply(y, goal)
        correctness = (result > 0)
        while not correctness.all():
            where = np.where(result <= 0)
            i = random.choice(where[0].tolist())
            grad = self.gram(x).sum(axis=1)
            alpha[i] = alpha[i] - learning_rate * (-grad[i])
            b = b - learning_rate * (-y[i])
            goal = np.multiply(alpha*y, self.gram(x)).sum(axis=1) + b
            result = np.multiply(y, goal)
            correctness = (result > 0)
        prediction = np.frompyfunc(sign, 1, 1)(goal)
        self.alpha = alpha
        self.b = b
        self.prediction = prediction
        return self

if __name__ == '__main__':
    data = pd.DataFrame({"field1":[3,4,1],
                         "field2":[3,3,1],
                         "value":[1,1,-1]})
    X = data[["field1","field2"]]
    Y = data["value"]
    model = Perceptron()
    params = model.fit(x=X, y=Y, w0=[0,0], b0=0, learning_rate=0.1)
    print("Model:")
    print("w=", end="")
    print(model.w)
    print("b=", end="")
    print(model.b)
    print("prediction_results:", end="")
    print(model.prediction)
    print("\n", end="")
    model_dual = PerceptronDual()
    params_dual = model_dual.fit(x=X, y=Y, alpha0=[0,0,0,0], b0=0, learning_rate=0.1)
    print("Dual model:")
    print("alpha=", end="")
    print(model_dual.alpha)
    print("b=", end="")
    print(model_dual.b)
    print("prediction_results:", end="")
    print(model_dual.prediction)
