import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

pretrained_net = torchvision.models.resnet18(pretrained=True) #pretrained打开可以拿到权重
print(list(pretrained_net.children())[-3:])

net = nn.Sequential(*list(pretrained_net.children())[:-2])
X = torch.rand(size=(1, 3 , 320, 480))
print(net(X).shape)

##构造fcn模型
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))

#转置卷积层  10,15 => 320,480 (320 - 64 + 2*p + s / s = 10) 480 - 64 + 2p + s / s = 15
net.add_module(
    'transpose_conv',
    nn.ConvTranspose2d(
        num_classes,
        num_classes,
        kernel_size=64,
        padding=16,
        stride=32)
)

#转置卷积初始化