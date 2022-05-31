# (tfconda)
from UDLv3 import udl
from datasets_registration import datasetDict
import os
import numpy as np
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0, camSys2CamPerspSys0
from codes_py.toolbox_bbox.cxcyrxry_func_v1 import croppingCxcyrxry0
from codes_py.np_ext.np_image_io_v1 import imread
import cv2
import matplotlib.pyplot as plt
from Bpredwood2.testDataEntry.scannetGivenRenderDataset import ScannetGivenRenderDataset
from Bpredwood2.testDataEntry.omnidataBerkeley.omnidataBerkeley \
    import OmnidataBerkeleyDataset


bt = lambda s: s[0].upper() + s[1:]


class PScannetGivenRenderDataset(ScannetGivenRenderDataset):
    pass


class POmnidataBerkeleyDataset(OmnidataBerkeleyDataset):
    pass
