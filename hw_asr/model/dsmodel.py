from torch import nn
import torch.nn.functional as F

from hw_asr.base import BaseModel


class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """inspired by https://arxiv.org/pdf/1603.05027.pdf
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super().__init__()
        self.do_residual = in_channels != out_channels
        if self.do_residual:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.net = nn.Sequential(
            CNNLayerNorm(n_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel//2),
            CNNLayerNorm(n_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2)
        )

    def forward(self, x):
        if self.do_residual:
            residual = self.residual(x)
        else:
            residual = x
        x = self.net(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first=True):
        super().__init__()
        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class DeepSpeechModel(BaseModel):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, kernel_size=3, dropout=0.1):
        super(DeepSpeechModel, self).__init__(n_feats, n_class)
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, kernel_size=3, stride=stride, padding=kernel_size // stride)

        layers = []
        for _ in range(n_cnn_layers):
            layers.append(ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats))
        self.cnn_net = nn.Sequential(*layers)
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)

        layers = [BidirectionalGRU(rnn_dim=rnn_dim, hidden_size=rnn_dim, dropout=dropout)]
        for _ in range(n_rnn_layers - 1):
            layers.append(BidirectionalGRU(rnn_dim=rnn_dim*2, hidden_size=rnn_dim, dropout=dropout))
        self.rnn_net = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        x = spectrogram.transpose(1, 2).unsqueeze(1)
        x = self.cnn(x)
        x = self.cnn_net(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.rnn_net(x)
        x = self.classifier(x)
        return x

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
