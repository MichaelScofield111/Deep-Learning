import matplotlib.pyplot as plt
import random
import  torch
from  d2l import torch as d2l


#data w = [2, -3.4]  b = 4.2
def syntheic_data(w, b, num_examples):
    """"y = Xw + b"""
    X = torch.normal(0,1,(num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = syntheic_data(true_w,true_b, 1000)
print('features', features[0], '\nlabels', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),
                labels.detach().numpy(),1)
plt.show()

#读取一个小批量batch_size;
def data_iter(batch_size, features, labels):
    """len是读取第一维度的大小"""
    num_examples = len(features)
    """获得索引"""
    indices = list(range(num_examples))
    """打乱一下索引顺序"""
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
    """yield 是一个关键字，在 Python 中用于生成迭代器的函数。这个关键字可以用于定义一个函数作为一个生成器。"""
    yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

### init model
w = torch.normal(0, 0.01, size=(2, 1),requires_grad=True) #需要求梯度
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    return torch.matmul(X, w) + b

### init loss function
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

### init sgd
def sgd(params, lr, batch_size):
    """"小批量梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


### traning
lr = 0.05
num_epochs = 30 #训练30遍
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        #求 w, b的梯度标量
        l.sum().backward()
        #update w and b
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1}), loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

