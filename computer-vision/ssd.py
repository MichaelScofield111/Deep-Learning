import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

#num_inputs 输入的通道数 num_anchors锚框数 num_classes类别数
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)  #+1有一个背景类

def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs,
                     num_anchors * 4,
                     kernel_size=3, padding=1)
