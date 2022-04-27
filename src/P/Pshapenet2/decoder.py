# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com

# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn.functional import conv3d, conv_transpose3d
from UDLv3 import udl
import numpy as np
from collections import OrderedDict
from .batch_renorm import BatchRenorm
from codes_py.toolbox_3D.rotations_v1 import camSys2CamPerspSys0


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

        config = kwargs['config']
        # numFinalChannel = config.networkTwoNumFinalChannel
        # zzUsedKeys = config.zzUsedKeys
        numFinalChannel = 1
        zzUsedKeys = ['z6']

        relu, conv3d, conv3d_t = nn.ReLU, nn.Conv3d, nn.ConvTranspose3d
        bn = lambda v: BatchRenorm(v, eps=0.001)

        c_dim_recorder = {}
        if_def_layer = {}
        for i in range(7):
            if_def_layer['y%d' % i] = any([z in zzUsedKeys for z in ['z%d' % j for j in range(i, 7)]])

        if if_def_layer['y0']:
            self.stage0 = nn.Linear(2048, 64)
            c_dim_recorder['z0'] = 64

        if if_def_layer['y1']:
            self.stage1 = nn.Sequential(OrderedDict(
                r1=relu(), b1=bn(64), t1=conv3d_t(64, 256, 4, 4, 0),
            ))
            c_dim_recorder['z1'] = 256

        if if_def_layer['y2']:
            self.stage2 = nn.Sequential(OrderedDict(
                r1=relu(), b1=bn(256), c1=conv3d(256, 256, 3, 1, 1),
                r2=relu(), b2=bn(256),
                t1=conv3d_t(256, 128, 3, 2, 1, output_padding=1)
            ))
            c_dim_recorder['z2'] = 128

        if if_def_layer['y3']:
            in3c = 224
            self.stage3 = nn.Sequential(OrderedDict(
                r1=relu(), b1=bn(in3c), c1=conv3d(in3c, 128, 5, 1, 2),
                r2=relu(), b2=bn(128),
                t1=conv3d_t(128, 64, 7, 2, 3, output_padding=1)
            ))
            c_dim_recorder['z3'] = 64

        if if_def_layer['y4']:
            in4c = 112
            self.stage4 = nn.Sequential(OrderedDict(
                r1=relu(), b1=bn(in4c), c1=conv3d(in4c, 64, 5, 1, 2),
                r2=relu(), b2=bn(64),
                t1=conv3d_t(64, 32, 7, 2, 3, output_padding=1)
            ))
            c_dim_recorder['z4'] = 32

        if if_def_layer['y5']:
            in5c = 56
            self.stage5 = nn.Sequential(OrderedDict(
                r1=relu(), b1=bn(in5c), c1=conv3d(in5c, 32, 5, 1, 2),
                t2=relu(), b2=bn(32),
                t1=conv3d_t(32, 16, 7, 2, 3, output_padding=1)
            ))
            c_dim_recorder['z5'] = 16

        if if_def_layer['y6']:
            in6c = 28
            self.stage6 = nn.Sequential(OrderedDict(
                r1=relu(), b1=bn(in6c), c1=conv3d(in6c, 16, 5, 1, 2),
                r2=relu(), b2=bn(16),
                t1=conv3d_t(16, numFinalChannel, 7, 2, 3, output_padding=1)
            ))
            c_dim_recorder['z6'] = numFinalChannel

        # assert list(c_dim_recorder.keys()) == zzUsedKeys  This assertion is wrong.

        meta = {}
        meta['c_dim_recorder'] = c_dim_recorder
        meta['if_def_layer'] = if_def_layer
        self.meta = meta

    def forward(self, tt):
        if_def_layer = self.meta['if_def_layer']

        out = {}
        if if_def_layer['y0']:
            y0 = self.stage0(tt['ta'])
            out['y0'] = y0
        if if_def_layer['y1']:
            x = y0
            x = x[:, :, None, None, None]
            y1 = self.stage1(x)
            out['y1'] = y1
        if if_def_layer['y2']:
            x = y1
            y2 = self.stage2(x)
            out['y2'] = y2
        if if_def_layer['y3']:
            x = torch.cat([y2, tt['t5']], 1)
            y3 = self.stage3(x)
            out['y3'] = y3
        if if_def_layer['y4']:
            x = torch.cat([y3, tt['t4']], 1)
            y4 = self.stage4(x)
            out['y4'] = y4
        if if_def_layer['y5']:
            x = torch.cat([y4, tt['t3']], 1)
            y5 = self.stage5(x)
            out['y5'] = y5
        if if_def_layer['y6']:
            x = torch.cat([y5, tt['t2']], 1)
            y6 = self.stage6(x)
            out['y6'] = y6

        return out

