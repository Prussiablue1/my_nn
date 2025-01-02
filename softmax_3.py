# by 诚赤

import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l




# 数据预处理
trans = transforms.ToTensor()  # 将图片转换为Tensor，并归一化为[0, 1]区间
mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

# 创建数据加载器
batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False)





# 定义Softmax回归模型
class SoftmaxRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)  # 定义线性层

    def forward(self, x):
        return torch.softmax(self.fc(x.reshape((-1, self.fc.in_features))), dim=1)

# 设置超参数
input_size = 28 * 28  # MNIST图片大小 28x28
num_classes = 10  # MNIST有10个类别
lr = 0.1  # 学习率
num_epochs = 25  # 训练轮次

# 初始化模型、损失函数和优化器
net = SoftmaxRegression(input_size, num_classes)
loss_fn = nn.CrossEntropyLoss()  # 使用交叉熵作为损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=lr)




# 训练模型
def train(net, train_iter, loss_fn, optimizer, num_epochs):
    for epoch in range(num_epochs):
        net.train()  # 设置模型为训练模式
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_iter:
            optimizer.zero_grad()  # 清空梯度
            y_hat = net(X)  # 前向传播
            loss = loss_fn(y_hat, y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item()  # 累加损失
            predicted = y_hat.argmax(dim=1)  # 获取预测标签
            correct += (predicted == y).sum().item()  # 计算正确预测数
            total += y.size(0)  # 计算样本总数

        avg_loss = total_loss / len(train_iter)
        accuracy = correct / total * 100
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # 打印权重和偏置
        print(f'  Weight (w): {net.fc.weight.data[:5]}')  # 打印前5个权重值
        print(f'  Bias (b): {net.fc.bias.data[:5]}')  # 打印前5个偏置值

        if accuracy >= 90:
            print(f'Epoch {epoch} break!')
            break



# 评估模型
def evaluate(net, test_iter):
    net.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for X, y in test_iter:
            y_hat = net(X)
            predicted = y_hat.argmax(dim=1)  # 获取预测标签
            correct += (predicted == y).sum().item()  # 计算正确预测数
            total += y.size(0)
    accuracy = correct / total * 100
    print(f'Test Accuracy: {accuracy:.2f}%')



# 训练和评估
train(net, train_iter, loss_fn, optimizer, num_epochs)
evaluate(net, test_iter)


