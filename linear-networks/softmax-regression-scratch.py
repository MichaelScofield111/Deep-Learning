import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_outputs,num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

X = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
#行sum                                  列sum
print(X.sum(0,keepdim=True),'\n', X.sum(1, keepdim=True))