"""
ResNeXt in PyTorch.
Got this from https://github.com/kuangliu/pytorch-cifar/

See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.decoder import Decoder
from utils.initialization import init_params

class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, img_dim=None, decoded_channel=3, num_classes=10):
        super(ResNeXt, self).__init__()
        
        """ We added this. """
        self.inference = False
        self.loss_fn_cls = nn.CrossEntropyLoss()
        self.loss_fn_aux = nn.MSELoss()

        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)

        """ We added this. """
        # The following representation of hidden_channel & hidden_dim is a hack...
        self.decoder = Decoder(hidden_channel=1024, hidden_dim=[n//4 for n in img_dim][::-1], decoded_channel=decoded_channel, img_dim=img_dim)
        init_params(self)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)

        """ We modified this. """
        out_pool = F.avg_pool2d(out, 8)
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

def ResNeXt29_2x64d(**kwargs):
    return ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64, **kwargs)

def ResNeXt29_4x64d(**kwargs):
    return ResNeXt(num_blocks=[3,3,3], cardinality=4, bottleneck_width=64, **kwargs)

def ResNeXt29_8x64d(**kwargs):
    return ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=64, **kwargs)

def ResNeXt29_32x4d(**kwargs):
    return ResNeXt(num_blocks=[3,3,3], cardinality=32, bottleneck_width=4, **kwargs)