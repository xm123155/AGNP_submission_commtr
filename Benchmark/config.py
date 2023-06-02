import torch

dataset = 'agnp'
percent = 70  # 10 40 or 70
model = 'gru'  # gru or lstm
train = False
batch_size = 1196
device = torch.device('cuda')
inp_dim = 12
out_dim = 1
mid_layers = 2
lr = 0.001
weight_decay = 0.000
decay = 1.
epoch = 5000
