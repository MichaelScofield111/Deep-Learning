import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    # 224 - 11 + 4 + 2  / 4  -> 54 - 3 + 2 / 2 => 26 * 26
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    #26 - 5 + 4 + 1 -> 26 - 3 + 2 / 2 => 12 * 12
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),

    # 12 - 3 + 2 / 2 = 5
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Flatten(),
    # 256 * 5 * 5 = 6400
    nn.Linear(6400, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(4096,4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(4096, 10)
)

X = torch.randn(1 , 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, "output shape:\t", X.shape)

batch_size = 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224) #拉大图片

lr, num_epochs = 0.02, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())



