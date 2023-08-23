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

#所以在不同尺度下预测生成的值也不同 所以要进行连接提高计算效率
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)  #把通道数放到最后 4D->2D

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

print(concat_preds([Y1,Y2]).shape)

#自己定义一个神经网络，为了在多个尺度下检测我们定义了类似与VVG的块
#每次让通道数不变 - 高宽减半
def down_sample_blk(in_channels,out_channels):
    blk = []
    for _ in range(2):
        #做两次
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2)) #高宽减半
    return nn.Sequential(*blk)

#通道数变为10 高宽减半torch.Size([2, 10, 10, 10])
print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)

#基础网络块
#让输入通道3 -> 16 -> 32 -> 64 高宽不断减半 256 -> 128 -> 64 -> 32
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        #每次把通道数扩大2倍 且高宽减半
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)

print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)



#一个是基本网络块，第二个到第四个是高和宽减半块，最后一个模块使用全局最大池将高度和宽度都降到1。
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:#stage 2-3
        blk = down_sample_blk(128, 128) #因为数据集太小没必要继续增加通道数了
    return blk


#对每个块定义前向计算
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X) #Y (feature map) 做了前向传播
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio) #生成锚框
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

#定义完整的模型
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128] #模型一共有5个stage的输入通道
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))  #得到feature_map
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                      num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))
    #前向传播
    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5  #有5个stage
        for i in range(5): #对于每个stage
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X,getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}')
            )

        anchors = torch.cat(anchors, dim=1) #所有锚框放在一起
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

net = TinySSD(num_classes=1) #类别数是1
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
# output anchors: torch.Size([1, 5444, 4]) 5544 = 5个stage下的所有anchors -- 4 = 一个anchros有4个点
# output class preds: torch.Size([32, 5444, 2]) 32 = 批量大小 -- 5544 = 多少个锚框(对于每一个锚框,预测两个值) -- 2 = 输入的类别数 + 1背景
# output bbox preds: torch.Size([32, 21776])  21176 = 5444 * 4 其实就是对于每一个锚框有4个预测值

batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

#用gpu训练
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)


#定义损失函数(重要)  评价函数
#类的预测损失就是交叉熵
cls_loss = nn.CrossEntropyLoss(reduction='none') #不用把loss加起来 reduction='none'
#锚框的预测损失
bbox_loss = nn.L1Loss(reduction='none') # |y_hat - y|



# output anchors: torch.Size([1, 5444, 4]) 5544 = 5个stage下的所有anchors -- 4 = 一个anchros有4个点
# output class_preds: torch.Size([32, 5444, 2]) 32 = 批量大小 -- 5544 = 多少个锚框(对于每一个锚框,预测两个值) -- 2 = 输入的类别数 + 1背景
# output bbox_preds: torch.Size([32, 21776])  21176 = 5444 * 4 其实就是对于每一个锚框有4个预测值
def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_mask):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    #-1 代表的是把批量大小维和锚框个数维放在一起->对于每一个锚框就是一个样本
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1) #cls对锚框分类的loss

    bbox = bbox_loss(bbox_preds * bbox_mask, bbox_labels * bbox_mask).mean(dim=1)
    return cls + bbox

#评估精度
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])

net = net.to(device)
for epoch in range(num_epochs):
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        anchors, cls_preds, bbox_preds = net(X)
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')



X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)

def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
        plt.show()

display(img, output.cpu(), threshold=0.9)
plt.show()
