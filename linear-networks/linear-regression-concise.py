import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(date_arrays, batch_size, is_train=True):
    """构造一个pytorch数据迭代器"""
    dataset = data.TensorDataset(*date_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))

#`nn`是神经网络的缩写
# 输入是2, 输出是1
net = nn.Sequential(nn.Linear(2, 1))  #net为list of layer 一层一层的

#由于只有一层,只需要初始化net[0]权重即可
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

#均方误差
loss = nn.MSELoss()

#training
trainer = torch.optim.SGD(net.parameters(), lr = 0.02)

#net自带模型的参数只需传入w,b即可
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')