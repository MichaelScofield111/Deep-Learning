import torch
from torch import nn
from d2l import torch as d2l

# 输入X = n * w  kernel = k * k  输出Y: (n - k + 1) * (w - k + 1)
#Y = X 卷积 W + b

def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))

class Conv2D(nn.Module):
    def __int__(self, kernel_size):
        super().__init__()
        self.weight = nn.parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    #前向运算
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
print(Y)

print(corr2d(X.t(), K))

#学习由X生成Y的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

X = X.reshape((1, 1, 6 ,8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2 #均方误差
    conv2d.zero_grad() #清除梯度
    l.sum().backward()
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch{i + 1}, loss {l.sum():.3f}')

print((conv2d.weight.data[0]))