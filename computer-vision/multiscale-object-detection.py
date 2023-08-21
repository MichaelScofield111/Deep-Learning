import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
print(w,h)

#display_anchors函数定义如下。 我们[在特征图（fmap）上生成锚框（anchors），每个单位（像素）作为锚框的中心]。 由于锚框中的 (𝑥,𝑦)
#轴坐标值（anchors）已经被除以特征图（fmap）的宽度和高度，因此这些值介于0和1之间，表示特征图中锚框的相对位置。
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    fmap = torch.zeros((1, 10, fmap_h, fmap_w)) #batch_size取1 通道数取10
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)


display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
plt.show()
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
plt.show()
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
plt.show()
