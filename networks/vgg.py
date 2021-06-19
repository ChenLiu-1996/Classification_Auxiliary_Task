"""
VGG11/13/16/19 in Pytorch.
Got this from https://github.com/kuangliu/pytorch-cifar/

"""

import torch
import torch.nn as nn
from networks.decoder import Decoder
from utils.initialization import init_params

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, img_dim=None, decoded_channel=3, num_classes=10):
        super(VGG, self).__init__()

        """ We added this. """
        self.inference = False
        self.loss_fn_cls = nn.CrossEntropyLoss()
        self.loss_fn_aux = nn.MSELoss()

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

        """ We added this. """
        # The following representation of hidden_channel & hidden_dim is a hack...
        self.decoder = Decoder(hidden_channel=512, hidden_dim=[n//32 for n in img_dim][::-1], decoded_channel=decoded_channel, img_dim=img_dim)
        init_params(self)

    def forward(self, x):
        out = self.features(x)

        """ We modified this. """
        out_flat = out.view(out.size(0), -1)
        output_cls = self.classifier(out_flat)

        """ We modified this. """
        if not self.inference:
            # Auxiliary Task
            output_aux = self.decoder(out)
            return output_cls, output_aux
        else:
            return output_cls

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
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
