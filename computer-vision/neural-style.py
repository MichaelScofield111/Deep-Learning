import  matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from d2l import  torch as d2l

d2l.set_figsize()
#read imaage
content_img = d2l.Image.open('../img/rainier.jpg')
d2l.plt.imshow(content_img)
plt.show()

style_img = d2l.Image.open('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img)
plt.show()

#normalizetion
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, img_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(rgb_mean, rgb_std)
    ])
    return transforms(img).unsqueeze(0) #升一个batch_size

#反者操作一遍
def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

#特征抽取 采用VGG19的模型
pretrained_net = torchvision.models.vgg19(pretrained=True)

#样式层-> 用来匹配样式  内容层 -> 越靠
style_layers, content_layers = [0, 5, 10, 19, 28], [25]

net = nn.Sequential(*[
    pretrained_net.features[i] for i in range(max(content_layers + style_layers) + 1)
])


#给定我要的样式层和内容层来抽特征
def extract_features(X, content_layers, style_layer):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layer:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

#get_contents函数对内容图像抽取内容特征
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    content_Y, _ = extract_features(content_X,content_layers,style_layers)
    return content_X, content_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, style_Y = extract_features(style_X, content_layers,style_layers)
    return style_X, style_Y


#定义损失函数 Y_hat是生成图片内容的损失
def content_loss(Y_hat, Y):
    return torch.square(Y_hat - Y.detach()).mean()

#如何匹配两个图片的样式 通道之前的统计分布是可以匹配上的
def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1] # n = w * h * batch_size
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (n * num_channels) #先做协方差在Normalize

def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()  #detach() 去除张量的梯度信息

#每个像素和他上下左右的像素绝对值不要差太多 图片比较平均
def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

#损失函数的权重
content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、风格损失和全变分损失
    #zip 元素逐个配对
    contents_l = [content_loss(Y_hat,Y) * content_weight for Y_hat, Y in zip(contents_Y_hat, contents_Y)]

    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]

    tv_l = tv_loss(X) * tv_weight
    #对所有损失函数求和
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


#训练的是图篇 不是卷积的权重
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))  #随机化一下图片

    def forward(self):
        return self.weight  #对weight算梯度然后更新

def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer


def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X



device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)