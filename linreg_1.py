# by 诚赤

import random

import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l



#生成数据集
def make_data(w,b,n):
    """生成y=Xw+b+噪声"""
    x=torch.normal(0,1,(n,len(w)))
    y=torch.matmul(x,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return x,y.reshape((-1,1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = make_data(true_w, true_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])

plt.figure(figsize=(8, 6))
plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
plt.show()



#读取数据集
# 定义一个数据迭代器函数
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 获取数据集的总样本数
    indices = list(range(num_examples))  # 创建一个索引列表，包含所有样本的索引
    random.shuffle(indices)  # 打乱样本的顺序
    for i in range(0, num_examples, batch_size):  # 以 batch_size 为步长，遍历数据
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])  # 获取当前批次的样本索引
        yield features[batch_indices], labels[batch_indices]  # 返回当前批次的特征和标签

# 假设 features 和 labels 已经定义好
batch_size = 10
for X, y in data_iter(batch_size, features, labels):  # 调用数据迭代器，迭代获取批次
    print(X, '\n', y)  # 打印当前批次的特征和标签
    break  # 这里只打印第一个批次.




#初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


#定义模型
def linreg(X, w, b):
    y= torch.matmul(X, w) + b
    return y



#定义损失函数
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


#优化算法
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    """见梯度下降公式，lr为学习率，batch_size为样本数"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()




#训练
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # l 是一个批次的损失，形状是(batch_size, 1)，我们通过sum()将其转化为一个标量
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

