import torch

X = torch.arange(4.0)
X

#计算y关于X的梯度之前我们需要一个地方来存梯度
X.requires_grad_(True)
X.grad #默认是None

y = 2 * torch.dot(X, X) #内积
print(y)
y.backward()
X.grad # = 4X

X.grad == 4 * X
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
X.grad.zero_()
y = X.sum() # y = X1 + X2 + X3 ... 然后对每个分量求梯度就是全1
y.backward()
print(X.grad)

#假设我的y不是标量怎么办呢？
#但当调用向量的反向计算时，我们通常会试图计算一批训练样本中每个组成部分的损失函数的导数。
# 这里(，我们的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和
X.grad
X.grad.zero_()
y = X * X #理论上他的backward()是一个矩阵
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
X.grad


X.grad.zero_()
y = X * X
u = y.detach() #u 不再是关于x的函数 而是直接变成常数值就是x*x
z = u * X

z.sum().backward()
print(X.grad == u)


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

a.grad == d / a