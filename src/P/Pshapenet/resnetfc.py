# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
import torch
from torch.nn.functional import softplus, relu

#  import torch_scatter
import torch.autograd.profiler as profiler


def forwardReLUGradTrack(x4):
    step = (x4[..., :1] > 0).float().detach()
    val = x4[..., :1] * step
    grad = x4[..., 1:] * step
    x4 = torch.cat([val, grad], -1)
    return x4


def forwardLinearGradTrack(module, x4):
    val = module(x4[..., 0])[..., None]
    grad = torch.matmul(module.weight, x4[..., 1:])
    x4 = torch.cat([val, grad], -1)
    return x4


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0, **kwargs):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
            raise NotImplementedError
        else:
            self.activation = nn.ReLU()
            self.forwardActivationGradTrack = forwardReLUGradTrack
        self.forwardLinearGradTrack = forwardLinearGradTrack

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x, forwardMode):
        if forwardMode in ['valOnly']:
            return self.forwardValOnly(x)
        elif forwardMode in ['gradTrack']:
            return self.forwardGradTrack(x)
        else:
            raise NotImplementedError('Unknown forwardMode: %s' % self.forwardMode)

    def forwardValOnly(self, x):
        with profiler.record_function("resblock"):
            net = self.fc_0(self.activation(x))
            dx = self.fc_1(self.activation(net))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx

    def forwardGradTrack(self, x4):
        with profiler.record_function('resblock'):
            net4 = self.forwardLinearGradTrack(self.fc_0, self.forwardActivationGradTrack(x4))
            dx4 = self.forwardLinearGradTrack(self.fc_1, self.forwardActivationGradTrack(net4))

            if self.shortcut is not None:
                x_s_4 = self.forwardLinearGradTrack(self.shortcut, x4)
            else:
                x_s_4 = x4

            return x_s_4 + dx4


class ResNetFC(nn.Module):
    def __init__(self, d_in, d_out, n_blocks, d_latent, d_hidden, beta, combine_layer,
                 **kwargs):
        super(ResNetFC, self).__init__()
        if d_in > 0:
            self.lin_in = nn.Linear(d_in, d_hidden)
            nn.init.constant_(self.lin_in.bias, 0.0)
            nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")

        self.lin_out = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.n_blocks = n_blocks
        self.d_latent = d_latent
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        self.combine_layer = combine_layer
        self.use_spade = False

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(d_hidden, beta=beta) for i in range(n_blocks)]
        )

        if d_latent != 0:
            n_lin_z = min(combine_layer, n_blocks)
            self.lin_z = nn.ModuleList(
                [nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)]
            )
            for i in range(n_lin_z):
                nn.init.constant_(self.lin_z[i].bias, 0.0)
                nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
            raise NotImplementedError
        else:
            self.activation = nn.ReLU()
            self.forwardActivationGradTrack = forwardReLUGradTrack

        self.forwardLinearGradTrack = forwardLinearGradTrack

    def forward(self, x, latent, forwardMode):
        if forwardMode in ['valOnly']:
            return self.forwardValOnly(x, latent)
        elif forwardMode in ['gradTrack']:
            return self.forwardGradTrack(x, latent)
        else:
            raise NotImplementedError('Unknown forwardMode: %s' % self.forwardMode)

    def forwardValOnly(self, x, latent):
        if self.d_in > 0:
            x = self.lin_in(x)
        else:
            x = torch.zeros(self.d_hidden, device=zx.device)
        z = latent

        for blkid in range(self.n_blocks):

            if self.d_latent > 0 and blkid < self.combine_layer:
                tz = self.lin_z[blkid](z)
                x = x + tz

            x = self.blocks[blkid](x, forwardMode='valOnly')
        out = self.lin_out(self.activation(x))
        return out

    def forwardGradTrack(self, x4, latent4):
        if self.d_in > 0:
            x4 = self.forwardLinearGradTrack(self.lin_in, x4)
        else:
            x4 = torch.zeros(self.d_hidden, 4, device=zx.device)
        z4 = latent4

        for blkid in range(self.n_blocks):

            if self.d_latent > 0 and blkid < self.combine_layer:
                tz4 = self.forwardLinearGradTrack(self.lin_z[blkid], z4)
                x4 = x4 + tz4

            x4 = self.blocks[blkid](x4, forwardMode='gradTrack')
        out4 = self.forwardLinearGradTrack(self.lin_out, self.forwardActivationGradTrack(x4))
        return out4
