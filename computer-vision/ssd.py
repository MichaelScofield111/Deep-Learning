import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


#有   (num_anchors * num_classes + 1)* num_h * num_w  =>把类别放进去
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)  #+1有一个背景类

#得到num_anchors * num_h * num_w的框每个框有4个位置 => num_anchors * 4 * num_h * num_w => 把每个锚框位置放进去
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs,
                     num_anchors * 4,
                     kernel_size=3, padding=1)

def forward(x,block):
    return block(x)

#torch.zeros((2, 8, 20, 20) feature map
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
print(Y1.shape, Y2.shape)

def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)  #把通道数放到最后

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

print(concat_preds([Y1,Y2]).shape)

#自己定义一个神经网络
def down_sample_blk(in_channels,out_channels):
    blk = []
    for _ in range(2):
        #做两次
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)