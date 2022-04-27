# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple

import torch
import torchvision
import torch.nn as nn
from scipy import signal
import numpy as np


class ResnetEncoder(torch.nn.Module):
    def __init__(self, requires_grad, requires_gaussian_blur, encoderTag,
                 ifNeedInternalRgbNormalization=True):
        super(ResnetEncoder, self).__init__()
        if encoderTag == 'resnet34':
            resnet = torchvision.models.resnet34(pretrained=True)
            c_dim = 64 + 64 + 128 + 256 + 512
        elif encoderTag == 'resnet50':
            resnet = torchvision.models.resnet50(pretrained=True)
            c_dim = 64 + 256 + 512 + 1024 + 2048
        elif encoderTag == 'resnet152':
            resnet = torchvision.models.resnet152(pretrained=True)
            c_dim = 64 + 256 + 512 + 1024 + 2048
        else:
            raise NotImplementedError('Unknown encoderTag: %s' % encoderTag)
        self.slice0 = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.slice1 = torch.nn.Sequential(
            resnet.maxpool,
            resnet.layer1,
        )
        self.slice2 = torch.nn.Sequential(
            resnet.layer2,
        )
        self.slice3 = torch.nn.Sequential(
            resnet.layer3,
        )
        self.slice4 = torch.nn.Sequential(
            resnet.layer4,
        )

        if True:
            self.add_module('gauLayer0', self.createGaussianFilter(5, 1))
            self.add_module('gauLayer1', self.createGaussianFilter(5, 0.9))
            self.add_module('gauLayer2', self.createGaussianFilter(3, 0.8))
            self.add_module('gauLayer3', self.createGaussianFilter(3, 0.7))
            self.add_module('gauLayer4', self.createGaussianFilter(3, 0.6))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if ifNeedInternalRgbNormalization:
            self.register_buffer(
                "_normalizer_mean_imagenet",
                torch.from_numpy(np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]))
            self.register_buffer(
                "_normalizer_std_imagenet",
                torch.from_numpy(np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]))

        self.requires_grad = requires_grad
        self.requires_gaussian_blur = requires_gaussian_blur
        self.ifNeedInternalRgbNormalization = ifNeedInternalRgbNormalization

        self.c_dim = c_dim

    @staticmethod
    def createGaussianFilter(size, sigma):
        gau = signal.gaussian(size, sigma)
        gau = gau / gau.sum()
        gau = np.outer(gau, gau)[None, None, :, :].astype(np.float32)  # Do conv2 in the batch dimension to save kernel storage
        gau = torch.from_numpy(gau)
        assert size % 2 == 1
        padding = size // 2
        gauLayer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size, stride=1,
                             padding=padding, padding_mode='replicate').requires_grad_(False)
        gauLayer.weight.copy_(gau)
        gauLayer.bias.fill_(0)
        return gauLayer

    @staticmethod
    def applyGaussian(feat, gauLayer):
        # feat: (b, c, h, w)
        # gau: (1, 1, size, size)
        # Do F.conv2d in the batch dimension
        c = feat.shape[1]
        h = feat.shape[2]
        w = feat.shape[3]
        return gauLayer(feat.view((-1, 1, h, w))).view((-1, c, h, w))

    def _normalize_imagenet(self, x_input):
        x = torch.div(x_input - self._normalizer_mean_imagenet, self._normalizer_std_imagenet)
        return x

    def forward(self, x):
        if self.ifNeedInternalRgbNormalization:
            x = self._normalize_imagenet(x)
        h0 = self.slice0(x)
        h1 = self.slice1(h0)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        out = [h0, h1, h2, h3, h4]

        return out

