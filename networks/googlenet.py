"""
Got this from https://github.com/kuangliu/pytorch-cifar/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.decoder import Decoder
from utils.initialization import init_params

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, img_dim=None, decoded_channel=3, num_classes=10):
        super(GoogLeNet, self).__init__()
        
        """ We added this. """
        self.inference = False
        self.loss_fn_cls = nn.CrossEntropyLoss()
        self.loss_fn_aux = nn.MSELoss()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)
        
        """ We added this. """
        # The following representation of hidden_channel & hidden_dim is a hack...
        self.decoder = Decoder(hidden_channel=1024, hidden_dim=[n//4 for n in img_dim][::-1], decoded_channel=decoded_channel, img_dim=img_dim)
        init_params(self)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)

        """ We modified this. """
        out_pool = self.avgpool(out)
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