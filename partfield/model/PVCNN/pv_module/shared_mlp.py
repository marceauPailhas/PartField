import torch.nn as nn

__all__ = ['SharedMLP']


class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1, device='cpu'):
        super().__init__()
        # print('==> SharedMLP device: ', device)
        if dim == 1:
            conv = nn.Conv1d
            bn = nn.InstanceNorm1d
        elif dim == 2:
            conv = nn.Conv2d
            bn = nn.InstanceNorm1d
        else:
            raise ValueError
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.extend(
                [
                    conv(in_channels, oc, 1, device=device),
                    bn(oc, device=device),
                    nn.ReLU(True),
                ])
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)
