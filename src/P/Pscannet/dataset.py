# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from UDLv3 import udl
from datasets_registration import datasetDict
import os
import numpy as np
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0, camSys2CamPerspSys0
from codes_py.toolbox_bbox.cxcyrxry_func_v1 import croppingCxcyrxry0
from codes_py.np_ext.np_image_io_v1 import imread
import cv2
import matplotlib.pyplot as plt
from Bpscannet.testDataEntry.scannetGivenRenderDataset import ScannetGivenRenderDataset


bt = lambda s: s[0].upper() + s[1:]


class PScannetGivenRenderDataset(ScannetGivenRenderDataset):
    pass
