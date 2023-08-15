import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

torch.set_printoptions(2)

#锚框的生成,对于每一个像素生成不同宽度和高度的锚框  data图片 sizes放缩比, ratios高宽比
def mutibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    in_height, in_width = data.shape[-2:]
    device, num_size, num_ratios = data.device, len(sizes),len(ratios)
    """每个像素点的锚框数量"""
    boxes_per_pixel = (num_size + num_ratios - 1) #每个像素的锚框数
    size_tensor = torch.tensor(sizes, device=device)#list -> tensor
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长
    steps_w = 1.0 / in_width  # 在x轴上缩放步长

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing="ij")
    # 这边举一个简单的例子 来看一下torch.meshgrid()函数的工作原理
    # center_h： tensor([0.1250, 0.3750, 0.6250, 0.8750])
    # center_w： tensor([0.1250, 0.3750, 0.6250, 0.8750])
    #  shift_y tensor([
    #               [0.1250, 0.1250, 0.1250, 0.1250],
    #               [0.3750, 0.3750, 0.3750, 0.3750],
    #               [0.6250, 0.6250, 0.6250, 0.6250],
    #               [0.8750, 0.8750, 0.8750, 0.8750]])

    # shift_x tensor([
    #               [0.1250, 0.3750, 0.6250, 0.8750],
    #               [0.1250, 0.3750, 0.6250, 0.8750],
    #               [0.1250, 0.3750, 0.6250, 0.8750],
    #               [0.1250, 0.3750, 0.6250, 0.8750]])
    #生成一维张量
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)#展平
    # 全部每个像素中心点坐标
    # tensor([0.1250, 0.1250, 0.1250, 0.1250, 0.3750, 0.3750, 0.3750, 0.3750, 0.6250, 0.6250, 0.6250, 0.6250, 0.8750, 0.8750, 0.8750, 0.8750])
    # tensor([0.1250, 0.3750, 0.6250, 0.8750, 0.1250, 0.3750, 0.6250, 0.8750, 0.1250, 0.3750, 0.6250, 0.8750, 0.1250, 0.3750, 0.6250, 0.8750])


    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    #计算所有锚框的w
    w = torch.cat((sizes[0] * torch.sqrt(in_height * ratio_tensor[:] / in_width), size_tensor[1:] * torch.sqrt(in_height * ratio_tensor[0] / in_width)))
    #计算锚框所有的h
    h = torch.cat((sizes[0] * torch.sqrt(in_width / in_height / ratio_tensor[:]), size_tensor[1:] * torch.sqrt(in_width / in_height / ratio_tensor[0])))
    # 除以2来获得半高和半宽(半高半宽和中心坐标相加后，就得到左上角和右下角的坐标)
    # 原本每个w的shape是[1，5], repeat 8000次。
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    # 所以最后整个anchor_manipulations的shpe就是[8000*5,4]


    #我们有了中心坐标和锚框的宽高，将两者相加得到左上角和右下角的坐标来表示所有锚框(一共有8000*5个锚框)，最终在output前面增加一个batch_size的维度，得到[1,40000,4]的输出。
    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x,shift_y,shift_x,shift_y],dim=1).repeat_interleave(boxes_per_pixel,dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)



img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = mutibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)
#torch.Size([1, 2042040, 4])  2042040 = 516 * 728 * 5锚框数量

boxes = Y.reshape(h, w, 5, 4)
print(boxes[250, 250, 0, :])



#显示以图像中以某个像素为中心的所有锚框
def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj,(list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors,['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        # 下面的是用来进入text文字描述的
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))



d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
plt.show()

#交并比（IoU）
def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    # boxes1(左上角x，左上角y，右下角x，右下角y)
    # 这边举个例子：boxes1 [1,1,3,3],[0,0,2,4],[1,2,3,4]]   ；boxes2[[0,0,3,3],[2,0,5,2]]
    # 首先计算一个框的面积（长X宽）
    box_area = lambda boxes: ((boxes[:,2] - boxes[:,0]) *
                               boxes[:,1] - boxes[:,3])
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # 重叠部分左上角坐标（取最大的值） 先提取左上角的坐标
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # boxes1：tensor([[1, 1, 3, 3],  ----->  [[[1, 1]],
    #                [0, 0, 2, 4],           [[0, 0]],
    #                [1, 2, 3, 4]])          [[1, 2]]]
    #
    # boxes2:tensor([[0, 0, 3, 3],   ----> [[0, 0],
    #               [2, 0, 5, 2]])         [2, 0]]
    #由于它们的维度不同，所以要用广播机制，真正计算的时候，是下面这样的
    #tensor([[[1, 1],[1, 1]],
    #        [[0, 0],[0, 0]],
    #       [[1, 2],[1, 2]]])
    #tensor([[0, 0],[2, 0]]
    #        [[0, 0],[2, 0]]
    #       [[0, 0],[2, 0]])
    # 此时inter_upperlefts 为：
    # tensor([[[1, 1],
    #         [2, 1]],
    #        [[0, 0],
    #        [2, 0]],
    #       [[1, 2],
    #        [2, 2]]])

    # 重叠部分左下角坐标（取最大的值），同上面的过程，就不再赘述了
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # clamp(min=0)用来限制inters最小不能低于0
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    # 这边又用了一次广播机制
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


