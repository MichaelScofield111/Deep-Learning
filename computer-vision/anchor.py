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
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
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



def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框。"""
    # 锚框数量和真实边界框数量
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0] #在例子中是5个anchors和2个gt
    # 计算每个锚框与每个真实边界框的iou值，[n1,n2]
    jaccard = box_iou(anchors, ground_truth)
    """
       tensor([[0.0536, 0.0000],
               [0.1417, 0.0000],
               [0.0000, 0.5657],
               [0.0000, 0.2059],
               [0.0000, 0.7459]])
       """
    # 定义anchors_bbox_map来记录anchor分别对应着什么gt，anchors_bbox_map存放标签初始全为-1
    anchors_bbox_map = torch.full((num_anchors), -1, dtype=torch.long, device=device)

    # 得到每行的最大值，即对于每个锚框来说，iou最大的那个真实边界框，返回iou值和对应真实边界框索引值[n1],[n1]
    max_ious, indices = torch.max(jaccard, dim=1)
    # 根据阈值得到锚框不为背景的相应的索引值[<=n1]
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    # 根据阈值得到锚框不为背景的真实边界框索引值[<=n1]，与anc_i一一对应的
    box_j = indices[max_ious >= iou_threshold]
    # 挑出>=iou_threshold的值,重新赋值，也就是对每个锚框，得到大于给定阈值的匹配的真实gt边界框的对应索引
    anchors_bbox_map[anc_i] = box_j
    # 行，列的默认值[n1],[n2]
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)

    #保证每个gt都有锚框,故开始重新分配锚框
    for _ in range(num_gt_boxes):
        # 取得该矩形中最大值的索引，是按reshape(-1)得到的索引 0-(n1*n2-1)
        max_idx = torch.argmax(jaccard)
        # 得到矩阵最大值所在的列，就是对应的真实gt边界框的索引
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()

        anchors_bbox_map[anc_idx] = box_idx
        # 将最大值所在该行置为-1
        jaccard[:, box_idx] = col_discard
        # 将最大值所在该列置为-1
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


#要好预测,尽量均值为0,分的比较开
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换。"""
    # 坐标转换 从（左上，右下）转换到（中间，宽度，高度）
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb) #真实边框坐标
    # 偏移量计算公式
    # 除0.2和0.1就是*10和*5
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(c_assigned_bb[:, 2:] / c_anc[:, 2:] + eps)

    #拼接
    offset = torch.cat([offset_xy, offset_wh], axis = 1)
    return offset



# anchors输入的锚框[1,锚框总数，4] labels真实标签[bn,真实锚框数，5]
#在下面的例子中为5，gt数目为2

# labels = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
#                          [1, 0.55, 0.2, 0.9, 0.88]])
# anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
#                     [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
#                     [0.57, 0.3, 0.92, 0.9]])

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框。"""
    batch_size, anchors = labels.shape[0], anchors.squeeze[0]
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    # 处理每个label
    for i in range(batch_size):
        #真实边框
        label = label[i, :, :]
        # 为每个锚框分配真实的边界框
        # assign_anchor_to_bbox函数返回，每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
        # tensor([-1,  0,  1, -1,  1])
        # 这边label[:, 1:] 从1开始是因为，求IOU的时候不需要用到类别
        anchors_bbox_map = assign_anchor_to_bbox(label[:,1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)

        # 将类标签和分配的边界框坐标初始化为零，tensor([0, 0, 0, 0, 0])
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        # 所有anchor对应的真实边框坐标
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        # 如果一个锚框没有被分配，我们标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)  #找到索引不是背景
        # 非背景对应的类别标签索引 0,1,1
        bb_idx = anchors_bbox_map[indices_true] #拿到标签了
        class_labels[indices_true] = label[bb_idx, 0].long() + 1 #把有标签的锚框，取成它的标签数+1
        assigned_bb[indices_true] = label[bb_idx, 1:]

        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)



ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
plt.show()

#预测的时候anchors是已知的，offset_preds是预测出来的
# 该函数将锚框和偏移量预测作为输入，并应用逆偏移变换来返回预测的边界框坐标。
def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框。"""
    # 从（左上，右下）转换到（中间，宽度，高度）
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    # 从（中间，宽度，高度）转换到（左上，右下）
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

# 按降序对置信度进行排序并返回其索引
def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim = -1, descending=True)
    keep = [] #保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)


#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)


anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])


output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
print(output)

fig = d2l.plt.imshow(img)
for i in output[0].detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)


plt.show()