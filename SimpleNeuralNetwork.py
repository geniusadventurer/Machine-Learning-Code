import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(x))

# 激活函数全部采用sigmoid函数
class Layer:
    def __init__(self, n_layers, *args):
        self.n_layers = n_layers
        self.n_nodes = []
        for arg in args:
            self.n_nodes.append(arg)
        self.value = [None] * n_layers
        self.w = [None] * (n_layers-1)
        self.b = [None] * (n_layers-1)
        for i in range(self.n_layers-1):
            self.w[i] = np.random.random_sample(size=(self.n_nodes[i], self.n_nodes[i+1]))
            self.b[i] = np.random.random_sample(size=(self.n_nodes[i+1], 1))
        self.dw = [None] * (n_layers-1)
        self.db = [None] * (n_layers-1)
        self.max_iter = 1000

    # 输入层
    def input_layer(self, *args):
        self.value[0] = np.array(args[0])
        return self

    # 前向传播（中间层与输出层）
    def forward(self, i, *args):
        if i != 0:
            self.value[i] = np.frompyfunc(sigmoid,1,1)(np.dot(self.w[i-1].T, self.value[i-1]) + self.b[i-1])
        return self

    # 神经网络模型的预测结果
    def prediction(self):
        return self.value[self.n_layers - 1]

    # 懒得推公式了，假定导数为下面的形式吧。
    def backward(self, i, learning_rate):
        self.dw[i] = np.log(self.value[i].astype(np.float))
        self.w[i] -= learning_rate * self.dw[i]
        self.db[i] = 1 / self.value[i+1] + np.log(self.value[i+1].astype(np.float))
        self.b[i] = self.b[i] - learning_rate * self.db[i]
        return self


if __name__ == '__main__':
    # 初始化4层神经网络（参数1），每层神经元数为3（输入）、4（隐藏[0]）、2（隐藏[1]）、4（输出）
    network = Layer(4,3,4,2,4)
    n = network.n_layers
    y_train = np.array([[1],[1],[0],[0]])
    # 创建输入层：参数为(-1,1)数组
    input_layer = network.input_layer([[2],[4],[6]])
    # 创建隐藏层：参数为在网络中的层次数
    hidden_layers = [None] * (n-2)
    hidden_layers[0] = network.forward(n-3)
    hidden_layers[1] = network.forward(n-2)
    # 创建输出层：只保留第1个参数
    output_layer = network.forward(n-1)
    # 迭代
    n_iter = 0
    while n_iter < 100:
        hidden_layers[1].backward(n-2, 0.7)
        hidden_layers[0].backward(n-3, 0.6)
        input_layer.backward(0, 0.7)
        hidden_layers[0].forward(n-3)
        hidden_layers[1].forward(n-2)
        output_layer.forward(n-1)
        if math.sqrt(np.sum(np.square(network.prediction() - y_train))) <= 0.01:
            break
        n_iter += 1
    print("迭代次数：", n_iter)
    print(network.prediction())
