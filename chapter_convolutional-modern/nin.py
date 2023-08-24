import torch
from torch import nn
from d2l import torch as d2l

# 全连接层可能带来过拟合
# 卷积层参数个数 = 输入的通道数 *   输出的通道数 * 窗口高宽
# 全连接参数个数 = 输入的通道 * 输入的高宽 * 输出的通道 *输出的高宽
#网络中的网络（NiN）提供了一个非常简单的解决方案：在每个像素的通道上分别使用多层感知机 (Lin et al., 2013)
#卷积层的输入和输出由四维张量组成，张量的每个轴分别对应样本、通道、高度和宽度。
# 另外，全连接层的输入和输出通常是分别对应于样本和特征的二维张量
#NiN的想法是在每个像素位置（针对每个高度和宽度）应用一个全连接层
#nin块
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
         nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )

#NiN模型
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2), #高宽减半
    nin_block(96,256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, 2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    #类别数是10 把通道压缩到10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten()
)

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)


lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())