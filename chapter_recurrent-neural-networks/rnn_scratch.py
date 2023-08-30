import matplotlib.pyplot as plt
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

#参数num_steps是每个子序列中预定义的时间步数
batch_size, num_steps = 32, 15

# X:  tensor([[27, 28, 29, 30, 31],
#         [ 2,  3,  4,  5,  6]])             train_iter
# Y: tensor([[28, 29, 30, 31, 32],
#         [ 3,  4,  5,  6,  7]])

#vocab 对象

train_iter, vocab = d2l.load_data_time_machine(batch_size,num_steps)

#返回[2,28]的shape
print(F.one_hot(torch.tensor([0, 2]), len(vocab)))

X = torch.arange(10).reshape((2, 5))
#做转置后把时间放在前面 目的访问的话可以连续
print(F.one_hot(X.T, 28).shape)

def get_params(vocab_size, num_hiddens, device):
    #输入是一个个词 one-hot后就是长是vocal_size的向量  输出其实就是做分类类别的个数也是vocal_size
    num_inputs = num_outputs = vocab_size

    #给我个shape生成均值为0方差为1的tensor
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens)) #上一个时刻的隐藏变量->下一个时刻的隐藏变量
    b_h = torch.zeros(num_hiddens, device=device)

    W_hq = normal((num_hiddens, num_outputs)) #隐藏变量到输出了w
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

#初始化隐藏状态 对于每一个样本隐藏状态是一个长为num_hiddens的向量
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), ) #初始的值都是0

#做计算如何在一个时间步内计算隐状态和输出
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    #input 序列的长度 批量大小 vocal_size
    for X in inputs: #沿这第一个维度去遍历它
        H = torch.tanh(torch.mm(X, W_xh)
                       + torch.mm(H, W_hh)
                       + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, )


class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

#example 样例
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
#X是一个2*5的矩阵
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
print(Y.shape, len(new_state), new_state[0].shape)


#预测函数 -> 给一个句子的开头生成 prefix, num_preds需要生成多少个词, net模型， vecor预测值map真实值
#prefix 作为前缀接着往下写
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    #初始隐藏状态
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]] #在vocal对应的下标存在outputs里
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1)) #把最近预测的当输出
    #给我一段话我来预测
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds): # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1))) #argmax返回的是下标
    return ' '.join([vocab.idx_to_token[i] for i in outputs])

print(predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))

#梯度剪裁  防止梯度爆炸
def grad_clipping(net, theta):
    """梯度剪裁"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(
        torch.sum(
            (p.grad**2)) for p in params
        )
    )
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2) # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                for s in state:
                    s.detach_()

        y = Y.T.reshape(-1) #y拉成一个向量 把时间信息放前面
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
        return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())


net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)




