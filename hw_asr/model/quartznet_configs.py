configs = {
    "5x5":
    [
    {'out_channels': 256, 'repeat': 1, 'kernel_size': 33, 'stride': 2, 'dilation': 1, 'dropout': 0.2, 'residual': False, 'time_separable': True},

    {'out_channels': 256, 'repeat': 5, 'kernel_size': 33, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'time_separable': True},

    {'out_channels': 256, 'repeat': 5, 'kernel_size': 39, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'time_separable': True},

    {'out_channels': 512, 'repeat': 5, 'kernel_size': 51, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'time_separable': True},

    {'out_channels': 512, 'repeat': 5, 'kernel_size': 63, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'time_separable': True},

    {'out_channels': 512, 'repeat': 5, 'kernel_size': 75, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'time_separable': True},

    {'out_channels': 512, 'repeat': 1, 'kernel_size': 87, 'stride': 1, 'dilation': 2, 'dropout': 0.2, 'residual': False, 'time_separable': True},

    {'out_channels': 1024, 'repeat': 1, 'kernel_size': 1, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': False, 'time_separable': False}
    ]
}