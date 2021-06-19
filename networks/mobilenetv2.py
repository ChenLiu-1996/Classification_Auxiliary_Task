"""
MobileNetV2 in PyTorch.
Got this from https://github.com/kuangliu/pytorch-cifar/

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.decoder import Decoder
from utils.initialization import init_params

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, img_dim=None, decoded_channel=3, num_classes=10):
        super(MobileNetV2, self).__init__()

        """ We added this. """
        self.inference = False
        self.loss_fn_cls = nn.CrossEntropyLoss()
        self.loss_fn_aux = nn.MSELoss()

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

        """ We added this. """
        # The following representation of hidden_channel & hidden_dim is a hack...
        self.decoder = Decoder(hidden_channel=1280, hidden_dim=[n//8 for n in img_dim][::-1], decoded_channel=decoded_channel, img_dim=img_dim)
        init_params(self)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10

        """ We modified this. """
        out_pool = F.avg_pool2d(out, 4)
        out_flat = out_pool.view(out_pool.size(0), -1)
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
