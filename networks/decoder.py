import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# The point of the current architecture of this decoder is that it can be used in 
# multiple different classification networks without relying on their achitectures.
class Decoder(nn.Module):
    """ Simplistic fully-connected decoder """
    def __init__(self, hidden_channel, hidden_dim, decoded_channel, img_dim):
        super(Decoder, self).__init__()

        # Find out how many deconvolutions we need
        # img_dim: (W, H, C)
        # hidden_dim: (C, H, W)
        scale_w = img_dim[0] // hidden_dim[2]
        scale_h = img_dim[1] // hidden_dim[1]
        assert(scale_w == scale_h)
        print("Decoder scaling (output / hidden):", scale_w)
        self.num_deconv = int(math.log(scale_w, 2))

        # Note: decoded_channel is not necessarily == img_dim[2].
        # If we do reconstruction, then yes.
        # If we do Fourier, then decoded_channel = 2 (magnitude & phase).
        self.hidden_dim = hidden_dim
        self.img_dim = img_dim
        
        for layer_idx in range(self.num_deconv):
            if layer_idx < self.num_deconv - 1:
                setattr(self, "deconv_" + str(layer_idx), \
                    nn.ConvTranspose2d(hidden_channel, hidden_channel, \
                    kernel_size=3, stride=2, padding=1, output_padding=1)
                )
            else:
                setattr(self, "deconv_" + str(layer_idx), \
                    nn.ConvTranspose2d(hidden_channel, decoded_channel, \
                    kernel_size=3, stride=2, padding=1, output_padding=1)
                )

    def forward(self, x):
        for layer_idx in range(self.num_deconv):
            deconv = getattr(self, "deconv_" + str(layer_idx))
            x = F.relu(deconv(x))
        return x

# class NaiveCNN(nn.Module):
#     """ Basic CNN architecture. """
#     def __init__(self, img_dim, num_classes=10):
#         super(NaiveCNN, self).__init__()

#         self.inference = False
#         self.loss_fn_cls = nn.CrossEntropyLoss()
#         self.loss_fn_aux = nn.MSELoss()

#         num_conv_layers = 3    # todo: Maybe write it s.t. changing this automatically changes the definitions of layers?
#         num_init_chnnl = 2**5
#         kernel_size = 3
#         stride = 1             # default
#         padding = 1
#         dilation = 1           # default

#         dim = np.array(img_dim[0:2][::-1]) # dim = (Height, Width)
#         for _ in range(num_conv_layers):
#             """ Using formula from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html """
#             # Assuming kernel_size, stride, padding, dilation, are the same along W and H.
#             dim = (dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

#         self.output_cls_dim = dim[0] * dim[1] * 2 * num_init_chnnl

#         self.conv1 = nn.Conv2d(img_dim[2], num_init_chnnl, kernel_size, stride, padding, dilation)
#         self.conv2 = nn.Conv2d(num_init_chnnl, 2 * num_init_chnnl, kernel_size, stride, padding, dilation)
#         self.conv3 = nn.Conv2d(2 * num_init_chnnl, 2 * num_init_chnnl, kernel_size, stride, padding, dilation)
#         self.fc = nn.Linear(self.output_cls_dim, num_classes)

#         # The following representation of hidden_dim is a hack...
#         self.decoder = Decoder(hidden_channel=2 * num_init_chnnl, hidden_dim=img_dim[::-1], img_dim=img_dim)

#         # """ Parameter initialization """
#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#         #         nn.init.constant_(m.weight, 1)
#         #         nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         """ Run auxiliary task during train & validation, but not during inference/test. """

#         # Primary Task (Classification)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x_flat = x.view(-1, self.output_cls_dim)
#         output_cls = self.fc(x_flat)
        
#         if not self.inference:
#             # Auxiliary Task
#             output_aux = self.decoder(x)
#             return output_cls, output_aux
#         else:
#             return output_cls

#     def loss(self, output_cls, y, output_aux, gt_aux, aux_type, aux_weight):        
#         loss_cls = self.loss_fn_cls(output_cls, y)

#         if aux_type is not None:
#             loss_aux = aux_weight * self.loss_fn_aux(output_aux, gt_aux)
#         else:
#             loss_aux = 0.0
            
#         # print("classification loss {:.3f}, auxiliary loss {:.3f}".format(loss_cls, loss_aux))

#         return loss_cls + loss_aux

#     def inference_mode(self):
#         self.inference = True