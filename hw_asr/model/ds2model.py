from torch import nn
from hw_asr.base import BaseModel


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, n_feats, activation, dropout=0.):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride,
                      padding=(kernel[0] // 2, kernel[1] // 2)),
            nn.BatchNorm2d(n_feats),
            activation()
        ]
        if dropout != 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x  # (batch, channel, feature, time)


# class BidirectionalGRU(nn.Module):
#     def __init__(self, rnn_dim, hidden_size, dropout, activation, batch_first=True):
#         super().__init__()
#         self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=hidden_size, num_layers=1, batch_first=batch_first, bidirectional=True)
#         self.activation = activation()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         x, _ = self.BiGRU(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#         return x


class DeepSpeechModel2(BaseModel):
    def __init__(self, rnn_dim, n_class, n_feats, dropout=0.1, activation='ReLU'):
        super(DeepSpeechModel2, self).__init__(n_feats, n_class)
        if activation == 'ReLU':
            activate_fn = nn.ReLU
        elif activation == 'GELU':
            activate_fn = nn.GELU
        else:
            raise KeyError(f'Unsupported activation function {activation}')

        self.cnn_net = nn.Sequential(*[
            CNNBlock(1, 32, kernel=(11, 41), stride=(2, 2), dropout=dropout, n_feats=32, activation=activate_fn),
            CNNBlock(32, 32, kernel=(11, 21), stride=(2, 1), dropout=dropout, n_feats=32, activation=activate_fn)
        ])

        # layers = [BidirectionalGRU(rnn_dim=32*(n_feats//4), hidden_size=rnn_dim, dropout=dropout, activation=activate_fn)]
        # for _ in range(4):
        #     layers.append(BidirectionalGRU(rnn_dim=rnn_dim*2, hidden_size=rnn_dim, dropout=dropout, activation=activate_fn))
        # self.rnn_net = nn.Sequential(*layers)

        self.rnn_net = nn.GRU(input_size=32 * (n_feats // 4), hidden_size=rnn_dim, dropout=dropout, num_layers=5,
                              bidirectional=True, batch_first=True)
        self.rnn_act = activate_fn()

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),
            activate_fn(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        x = spectrogram.transpose(1, 2).unsqueeze(1)
        x = self.cnn_net(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feat, time)
        x = x.transpose(1, 2)
        x, _ = self.rnn_net(x)
        x = self.rnn_act(x)
        x = self.classifier(x)
        return x

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
