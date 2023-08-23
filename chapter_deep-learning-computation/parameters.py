import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

X = torch.rand(size=(2, 4))
print(net(X))

#访问层里的参数 Sequential其实就是list(python)
print(net[2].state_dict())
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)


#一次性访问所有参数
print(*[(name, param.shape) for name, param in net.named_parameters()])

print(net.state_dict()['2.bias'].data)


#内置初始化 m是一个module
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01) #_替换函数,不是返回值,而是直接替换m.weight
        nn.init.zeros_(m.bias)

net.apply(init_normal) #apply 对于net里面所有的layer调用一下init_noraml
print(net[0].weight.data[0], net[0].bias.data)

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

#可以定义不同层的初始化
net[0].apply(xavier)
print(net[0].weight.data[0])

#共享权重  share绑定 第二个层和第三个层绑定
shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared,
    nn.ReLU(),
    nn.Linear(8,1)
)

net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])


