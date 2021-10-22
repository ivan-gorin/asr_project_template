from torch import nn

from hw_asr.base import BaseModel
from hw_asr.model.quartznet_configs import configs


def conv_bn_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                  time_separable=False,
                  norm_eps=1e-3):
    layers = []
    if time_separable:
        layers.append(nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=False, dilation=dilation, groups=in_channels))
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0,
                                bias=False, groups=groups))
    else:
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=False, dilation=dilation, groups=groups))
    layers.append(nn.BatchNorm1d(out_channels, eps=norm_eps))
    if groups > 1:
        raise NotImplementedError
    return nn.Sequential(*layers)


class MainBlock(nn.Module):
    def __init__(self, repeat, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, dropout=0.2,
                 time_separable=False, residual=False):
        super().__init__()
        self.is_residual = residual
        if residual:
            self.residual = conv_bn_layer(in_channels, out_channels, kernel_size=1)
        layers = []
        if dilation > 1:
            padding = (kernel_size * dilation) // 2 - 1
        else:
            padding = kernel_size // 2
        for i in range(repeat):
            layers.append(conv_bn_layer(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                        time_separable))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)
        self.tail = nn.ReLU()

    def forward(self, x):
        result = self.net(x)
        if self.is_residual:
            result += self.residual(x)
        return self.tail(result)


class QuartzNetModel(BaseModel):
    def __init__(self, config, n_feats, n_class, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)

        if config not in configs:
            raise ValueError("QuartNet config not found")

        layers = []
        for block in configs[config]:
            groups = block.get('groups', 1)
            layers.append(
                MainBlock(repeat=block['repeat'],
                          in_channels=n_feats,
                          out_channels=block['out_channels'],
                          kernel_size=block['kernel_size'],
                          stride=block['stride'],
                          dilation=block['dilation'],
                          groups=groups,
                          dropout=block['dropout'],
                          time_separable=block['time_separable'],
                          residual=block['residual'])
            )
            n_feats = block['out_channels']

        self.net = nn.Sequential(*layers)
        self.tail = nn.Conv1d(1024, n_class, kernel_size=1, bias=True)

    def forward(self, spectrogram, *args, **kwargs):
        # print(spectrogram.shape, "SPECTROGRAM")
        out = self.net(spectrogram.transpose(1, 2))
        # print(self.tail(out).shape, 'TAIL')
        return self.tail(out).transpose(1, 2)

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
