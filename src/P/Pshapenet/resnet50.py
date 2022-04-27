# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com

# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is modified from https://github.com/google-research/corenet

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResNet50 feature extractor."""
import collections
from typing import NamedTuple
from typing import Tuple

import torch as t
from torch import nn

from . import batch_renorm


class ResNet50Features(NamedTuple):
  stage1_64x128x128: t.Tensor
  stage2_256x64x64: t.Tensor
  stage3_512x32x32: t.Tensor
  stage4_1024x16x16: t.Tensor
  stage5_2048x8x8: t.Tensor
  global_average_2048: t.Tensor


class ResNetBlock(nn.Module):
  def __init__(self):
    super().__init__()

  def init_weights(self, module: nn.Module) -> nn.Module:
    for m in module.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
      elif isinstance(m, batch_renorm.BatchRenorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    return module


class IdentityBlock(ResNetBlock):

  def __init__(self, in_channels: int, kernel_size: int,
               filters: Tuple[int, int, int],
               return_output_before_relu=False):
    super().__init__()

    f1, f2, f3 = filters
    self.out_channels = f3
    assert kernel_size % 2 == 1  # For padding to work correctly
    self.return_output_before_relu = return_output_before_relu

    self.op_a = self.init_weights(nn.Sequential(collections.OrderedDict(
        conv=nn.Conv2d(in_channels=in_channels, out_channels=f1, kernel_size=1),
        bn=batch_renorm.BatchRenorm(f1, eps=0.001))))
    self.op_b = self.init_weights(nn.Sequential(collections.OrderedDict(
        conv=nn.Conv2d(in_channels=f1, out_channels=f2, kernel_size=kernel_size,
                       padding=kernel_size // 2),
        bn=batch_renorm.BatchRenorm(f2, eps=0.001))))
    self.op_c = self.init_weights(nn.Sequential(collections.OrderedDict(
        conv=nn.Conv2d(in_channels=f2, out_channels=f3, kernel_size=1),
        bn=batch_renorm.BatchRenorm(f3, eps=0.001))))

  def forward(self, x: t.Tensor):
    input_tensor = x
    x = self.op_a(x).relu()
    x = self.op_b(x).relu()
    x = self.op_c(x) + input_tensor
    output_before_relu = x
    x = x.relu()
    if self.return_output_before_relu:
      return x, output_before_relu
    else:
      return x


class DownscaleBlock(ResNetBlock):
  def __init__(self, in_channels: int, kernel_size: int,
               filters: Tuple[int, int, int],
               stride: int = 2):
    super().__init__()

    f1, f2, f3 = filters
    self.out_channels = f3

    self.op_a = self.init_weights(nn.Sequential(collections.OrderedDict(
        conv=nn.Conv2d(in_channels=in_channels, out_channels=f1,
                       kernel_size=1, stride=stride),
        bn=batch_renorm.BatchRenorm(f1, eps=0.001))))
    self.op_b = self.init_weights(nn.Sequential(collections.OrderedDict(
        conv=nn.Conv2d(in_channels=f1, out_channels=f2, kernel_size=kernel_size,
                       padding=kernel_size // 2),
        bn=batch_renorm.BatchRenorm(f2, eps=0.001))))
    self.op_c = self.init_weights(nn.Sequential(collections.OrderedDict(
        conv=nn.Conv2d(in_channels=f2, out_channels=f3, kernel_size=1),
        bn=batch_renorm.BatchRenorm(f3, eps=0.001))))
    self.shortcut = self.init_weights(nn.Sequential(collections.OrderedDict(
        conv=nn.Conv2d(in_channels=in_channels, out_channels=f3,
                       kernel_size=1, stride=stride),
        bn=batch_renorm.BatchRenorm(f3, eps=0.001))))

  def forward(self, x: t.Tensor) -> t.Tensor:
    s = self.shortcut(x)
    x = self.op_a(x).relu()
    x = self.op_b(x).relu()
    x = (self.op_c(x) + s).relu()
    return x


class ResNet50FeatureExtractor(ResNetBlock):
  def __init__(self, **kwargs):
    super().__init__()

    config = kwargs['config']
    # uuUsedKeys = config.uuUsedKeys
    # zzUsedKeys = config.zzUsedKeys
    uuUsedKeys = []
    zzUsedKeys = ['z6']

    self.stage1 = self.init_weights(nn.Sequential(collections.OrderedDict(
        pad=nn.ZeroPad2d(3),
        conv=nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                       stride=2))))

    self.stage1_part2 = self.init_weights(nn.Sequential(collections.OrderedDict(
        bn=batch_renorm.BatchRenorm(64, eps=0.001),
        relu=nn.ReLU(),
        pad=nn.ZeroPad2d(padding=1),
        pool=nn.MaxPool2d(kernel_size=3, stride=2))))

    self.stage2 = nn.Sequential(collections.OrderedDict(
        a=DownscaleBlock(in_channels=64, kernel_size=3, filters=(64, 64, 256),
                         stride=1),
        b=IdentityBlock(in_channels=256, kernel_size=3,
                        filters=(64, 64, 256)),
        c=IdentityBlock(in_channels=256, kernel_size=3, filters=(64, 64, 256),
                        return_output_before_relu=True)))

    self.stage3 = nn.Sequential(collections.OrderedDict(
        a=DownscaleBlock(in_channels=256, kernel_size=3,
                         filters=(128, 128, 512)),
        b=IdentityBlock(in_channels=512, kernel_size=3,
                        filters=(128, 128, 512)),
        c=IdentityBlock(in_channels=512, kernel_size=3,
                        filters=(128, 128, 512)),
        d=IdentityBlock(in_channels=512, kernel_size=3,
                        filters=(128, 128, 512),
                        return_output_before_relu=True)))

    self.stage4 = nn.Sequential(collections.OrderedDict(
        a=DownscaleBlock(in_channels=512, kernel_size=3,
                         filters=(256, 256, 1024)),
        b=IdentityBlock(in_channels=1024, kernel_size=3,
                        filters=(256, 256, 1024)),
        c=IdentityBlock(in_channels=1024, kernel_size=3,
                        filters=(256, 256, 1024)),
        d=IdentityBlock(in_channels=1024, kernel_size=3,
                        filters=(256, 256, 1024)),
        e=IdentityBlock(in_channels=1024, kernel_size=3,
                        filters=(256, 256, 1024)),
        f=IdentityBlock(in_channels=1024, kernel_size=3,
                        filters=(256, 256, 1024),
                        return_output_before_relu=True)))

    self.stage5 = nn.Sequential(collections.OrderedDict(
        a=DownscaleBlock(in_channels=1024, kernel_size=3,
                         filters=(512, 512, 2048)),
        b=IdentityBlock(in_channels=2048, kernel_size=3,
                        filters=(512, 512, 2048)),
        c=IdentityBlock(in_channels=2048, kernel_size=3,
                        filters=(512, 512, 2048),
                        return_output_before_relu=True)))

    c_dim_recorder = {}
    if_def_layer = {}
    if_def_layer['s1'] = any([u in uuUsedKeys for u in ['u1', 'u2', 'u3', 'u4', 'u5']])
    if_def_layer['s2'] = any([u in uuUsedKeys for u in ['u2', 'u3', 'u4', 'u5']]) or \
        any([z in zzUsedKeys for z in ['z6']])
    if_def_layer['s3'] = any([u in uuUsedKeys for u in ['u3', 'u4', 'u5']]) or \
        any([z in zzUsedKeys for z in ['z6', 'z5']])
    if_def_layer['s4'] = any([u in uuUsedKeys for u in ['u4', 'u5']]) or \
        any([z in zzUsedKeys for z in ['z6', 'z5', 'z4']])
    if_def_layer['s5'] = any([u in uuUsedKeys for u in ['u5']]) or \
        any([z in zzUsedKeys for z in ['z6', 'z5', 'z4', 'z3']])
    if if_def_layer['s1']:
        c_dim_recorder['u1'] = 64
    if if_def_layer['s2']:
        self.compress2 = nn.Conv2d(256, 12, 1, 1, 0)  # used by 3d_stage5
        c_dim_recorder['u2'] = 12
    if if_def_layer['s3']:
        self.compress3 = nn.Conv2d(512, 24, 1, 1, 0)  # used by 3d_stage4
        c_dim_recorder['u3'] = 24
    if if_def_layer['s4']:
        self.compress4 = nn.Conv2d(1024, 48, 1, 1, 0)  # used by 3d_stage3
        c_dim_recorder['u4'] = 48
    if if_def_layer['s5']:
        self.compress5 = nn.Conv2d(2048, 96, 1, 1, 0)  # used by 3d_stage2
        c_dim_recorder['u5'] = 96

    # The following assertion is wrong, as it can be uuUsedKeys empty but if_def_layer has Trues.
    # assert list(c_dim_recorder.keys()) == uuUsedKeys

    # meta
    meta = {}
    meta['c_dim_recorder'] = c_dim_recorder
    meta['if_def_layer'] = if_def_layer
    self.meta = meta

  def forward(self, input_image: t.Tensor):
    if_def_layer = self.meta['if_def_layer']

    x = stage1 = self.stage1(input_image)
    x = self.stage1_part2(x)
    x, stage2 = self.stage2(x)
    x, stage3 = self.stage3(x)
    x, stage4 = self.stage4(x)
    x, stage5 = self.stage5(x)
    avg_pool = x.mean(dim=(2, 3))

    # return ResNet50Features(stage1, stage2, stage3, stage4, stage5,
    #                         avg_pool)

    # s1 = stage1  # (B, 64, 128, 128)  # not used
    # s2 = self.compress2(stage2)  # (B, 12, 64, 64)  # used by 3d_stage5
    # s3 = self.compress3(stage3)  # (B, 24, 32, 32)  # used by 3d_stage4
    # s4 = self.compress4(stage4)  # (B, 48, 16, 16)  # used by 3d_stage3
    # s5 = self.compress5(stage5)  # (B, 96, 8, 8)  # used by 3d_stage2
    # sa = avg_pool  # (B, 2048)  # used by 3d_stage0
    out = {}
    out['sa'] = avg_pool
    out['s1'] = stage1
    if if_def_layer['s2']:
        out['s2'] = self.compress2(stage2)
    if if_def_layer['s3']:
        out['s3'] = self.compress3(stage3)
    if if_def_layer['s4']:
        out['s4'] = self.compress4(stage4)
    if if_def_layer['s5']:
        out['s5'] = self.compress5(stage5)
    return out


def preprocess_image_caffe(image: t.Tensor) -> t.Tensor:
  """Preprocess an image for use with Caffe's ImageNet ResNet weights.

  Args:
    image: uint8[batch, 3, height, width]

  Returns:
    Preprocessed image, float32[batch, 3, height, width]
  """
  assert (image.dtype == t.uint8 and len(image.shape) == 4 and
          image.shape[1] == 3)
  image = image.to(t.float32)
  image = image.flip(1)  # RGB->BGR
  image = (image +
           image.new_tensor([103.939, 116.779, 123.68])[None, :, None, None])
  return image
