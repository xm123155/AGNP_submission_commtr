from torch import nn


class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x):
        x = self.rnn(x)[0]
        seq_len, batch_size, hid_dim = x.shape
        x = x.view(-1, hid_dim)
        x = self.reg(x)
        x = x.view(seq_len, batch_size, -1)
        return x


class GRU(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim):
        super(GRU, self).__init__()
        self.inp_dim, self.out_dim, self.mid_dim = inp_dim, out_dim, mid_dim
        self.gru = nn.GRU(inp_dim, mid_dim)
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x):
        x, hidden = self.gru(x)
        x = self.reg(x)
        return x
