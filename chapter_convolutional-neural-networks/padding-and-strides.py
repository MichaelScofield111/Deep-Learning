import torch
from torch import nn

def comp_conv2d(con2d, X):
    X = X.reshape((1 , 1, X.shape[0], X.shape[1]))
    Y = con2d(X)
    return Y.reshape(Y.shape[2:])

#计算公式 Nh-Kh+(Ph*2) - 1 = 8 - 3 + 2 + 1  同理... 另个一也是8
con2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.randn(size=(8, 8))
print(comp_conv2d(con2d,X).shape)

# 8 - 5 + 4 + 1 = 8, 8 - 3 + 2 + 1 = 8
con2d = nn.Conv2d(1, 1, kernel_size=(5,3), padding=(2, 1))
print(comp_conv2d(con2d,X).shape)

#将步幅设置成2
# (8 - 3 + 2 + 2 ) / 2 = 4 向下取整 同理...
con2d = nn.Conv2d(1, 1 ,kernel_size=3, padding=1, stride=2)
print(comp_conv2d(con2d,X).shape)

# (8 - 3 + 0 + 3) / 3 = 2 , (8 - 5 + 2 + 4) / 4 = 2
con2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3 ,4))
print(comp_conv2d(con2d,X).shape)