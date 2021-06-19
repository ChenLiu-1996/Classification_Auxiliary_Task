"""
Got this from https://github.com/kuangliu/pytorch-cifar/

Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".

Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.decoder import Decoder
from utils.initialization import init_params

def swish(x):
    return x * x.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, img_dim=None, decoded_channel=3, num_classes=10):
        super(EfficientNet, self).__init__()

        """ We added this. """
        self.inference = False
        self.loss_fn_cls = nn.CrossEntropyLoss()
        self.loss_fn_aux = nn.MSELoss()

        self.cfg = cfg
        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(cfg['out_channels'][-1], num_classes)

        """ We added this. """
        # The following representation of hidden_channel & hidden_dim is a hack...
        self.decoder = Decoder(hidden_channel=cfg['out_channels'][-1], hidden_dim=[n//16 for n in img_dim][::-1], decoded_channel=decoded_channel, img_dim=img_dim)
        init_params(self)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
                                     'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block(in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)

        """ We modified this. """
        out_pool = F.adaptive_avg_pool2d(out, 1)
        dropout_rate = self.cfg['dropout_rate']
        out_flat = out_pool.view(out_pool.size(0), -1)
        if self.training and dropout_rate > 0:
            out_flat = F.dropout(out_flat, p=dropout_rate)
        output_cls = self.linear(out_flat)

        """ We modified this. """
        if not self.inference:
            # Auxiliary Task
            output_aux = self.decoder(out)
            return output_cls, output_aux
        else:
            return output_cls

    """ We added this. """
    def loss(self, output_cls, y, output_aux, gt_aux, aux_type, aux_weight):        
        loss_cls = self.loss_fn_cls(output_cls, y)

        if aux_type is not None:
            loss_aux = aux_weight * self.loss_fn_aux(output_aux, gt_aux)
        else:
            loss_aux = 0.0

        return loss_cls + loss_aux

    """ We added this. """
    def non_inference_mode(self):
        self.inference = False

    """ We added this. """
    def inference_mode(self):
        self.inference = True


def EfficientNetB0(**kwargs):
    cfg = {
        'num_blocks': [1, 2, 2, 3, 3, 4, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'out_channels': [16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
    }
    return EfficientNet(cfg, **kwargs)