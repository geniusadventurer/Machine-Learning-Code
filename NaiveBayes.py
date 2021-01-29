import pandas as pd

# 这里的思路是用pandas里按条件取数的功能
class NaiveBayes:
    # 模型拟合，lbd为贝叶斯公式中的λ，因为计算公式里有y所以把x和y作为整体输入
    def fit(self, train_data, test_data, lbd):
        m = test_data.shape[0] # 待分类样本数为m个
        n = test_data.shape[1] # 变量数为n个
        S = []
        for i in range(n):
            S.append(len(train_data.iloc[:,i].unique())) # 求x的唯一值的数目
        y_unique = train_data.iloc[:,-1].unique() # 获得y的唯一值
        K = len(y_unique) # 求y的唯一值的数目
        prior_y = {}
        for y in y_unique:
            prior_y[y] = (len(train_data[train_data.iloc[:,-1]==y]) + lbd) / (len(train_data) + K * lbd) # 求每个y值对应的先验概率
        results = []
        for j in range(m):
            p_lst = {}
            for y in y_unique:
                p = prior_y[y] # 先求先验概率
                for i in range(n):
                    n1 = len(train_data[(train_data.iloc[:,i]==test_data.iloc[j,i]) & (train_data.iloc[:,-1]==y)]) + lbd # 再依次求条件n的概率
                    p = p * (n1 / (len(train_data[train_data.iloc[:,-1]==y]) + S[i] * lbd)) # 连乘
                    p_lst[y] = p # 将乘出来得到的概率放入p_lst
            p_max = max(p_lst.values())
            for key,value in p_lst.items():
                if value == p_max:
                    results.append(key) # 把概率最大的值对应的y放入结果列表
        self.result = results # 获取结果
        return self

if __name__ == '__main__':
    data = pd.DataFrame({"field1":[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
                         "field2":["S","M","M","S","S","S","M","M","L","L","L","M","M","L","L"],
                         "value":[-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]})
    test = pd.DataFrame({"field1":[1,2,3],
                         "field2":["S","S","M"]})
    model = NaiveBayes().fit(train_data=data, test_data=test, lbd=1)
    print("prediction:")
    print(model.result)
