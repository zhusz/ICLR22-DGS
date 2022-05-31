# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (tfconda)
import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
sys.path.append(projRoot + 'src/B/')
from BpredwoodA.testDataEntry.hmRenderOmDataset import HmMeshCacheFromRaw
from UDLv3 import udl
from codes_py.toolbox_3D.rotations_v1 import quat2matNP, ELU02cam0
from codes_py.toolbox_3D.mesh_io_v1 import load_obj_np
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_show_draw.draw_v1 import to_heatmap
import math
import numpy as np
import json
from skimage.io import imread
import trimesh
import pickle
from codes_py.toolbox_3D.pyrender_wrapper_v1 import PyrenderManager
from collections import OrderedDict
from easydict import EasyDict


def main():
    dataset = os.path.realpath(__file__).split('/')[-2]
    rgb_raw_root = projRoot + 'remote_fastdata/omnidata/om/rgb/hm3d/'  # subject to change
    point_info_raw_root = projRoot + 'remote_fastdata/omnidata/om/point_info/hm3d/'  # subject to change
    zbuffer_raw_root = projRoot + 'remote_fastdata/omnidata/om/depth_zbuffer/hm3d/'

    dataset_house = 'hm'
    assert dataset.startswith(dataset_house)
    A1_house = udl('pkl_A1_', dataset_house)
    m_house = A1_house['m']
    houseShaList_house = A1_house['houseShaList']
    flagSplit_house = udl('_A01_flagSplit', dataset_house)

    A1 = udl('pkl_A1_', dataset)
    houseIDList = A1['houseIDList']
    pointIDList = A1['pointIDList']
    viewIDList = A1['viewIDList']
    EWorld = A1['EWorld']
    LWorld = A1['LWorld']
    UWorld = A1['UWorld']
    resolution = A1['resolution']
    fScaleList = A1['fScaleList']

    flagSplit = udl('_A01_flagSplit', dataset)

    indVisChosen = [8470854]

    pyrenderManager = PyrenderManager(512, 512)
    assert os.path.isdir(projRoot + 'cache/')
    visualDir = projRoot + 'cache/dataset/%s/play_A1/' % dataset
    os.makedirs(visualDir, exist_ok=True)
    htmlStepper = HTMLStepper(visualDir, 100, 'play_A1')

    hmMeshCache = HmMeshCacheFromRaw(zNear=1.e-6)

    for j in indVisChosen:
        houseID = int(houseIDList[j])
        pointID = int(pointIDList[j])
        viewID = int(viewIDList[j])
        EWorld0 = EWorld[j, :]
        LWorld0 = LWorld[j, :]
        UWorld0 = UWorld[j, :]
        resolution0 = float(resolution[j])
        assert resolution0 == 512
        fScale0 = float(fScaleList[j])

        curr_rgb_raw_dir = rgb_raw_root + '%05d-' % houseID + \
                           houseShaList_house[houseID] + '/'
        curr_zbuffer_raw_dir = zbuffer_raw_root + '%05d-' % houseID + \
                               houseShaList_house[houseID] + '/'
        curr_point_info_raw_dir = point_info_raw_root + '%05d-' % houseID + \
                                  houseShaList_house[houseID] + '/'

        summary0 = OrderedDict([])
        txt0 = []
        brInds = [0, ]

        omnidata_img = imread(
            curr_rgb_raw_dir + 'point_%d_view_%d_domain_rgb.png' % (pointID, viewID))
        summary0['omnidata_img_%d_%d' % (pointID, viewID)] = omnidata_img
        txt0.append('Point %d View %d img' % (pointID, viewID))

        omnidata_depth = imread(
            curr_zbuffer_raw_dir + 'point_%d_view_%d_domain_depth_zbuffer.png' %
            (pointID, viewID))

        omnidata_depth = omnidata_depth.astype(np.float32) / 512.  # only for omnidata-hm3d
        # other datasets under omnidata is 8000 for this one.

        summary0['omnidata_depthZBuffer_%d_%d' % (
            pointID, viewID)] = to_heatmap(omnidata_depth, vmin=0.5, vmax=6, cmap='inferno')
        txt0.append('Point %d View %d depthZBuffer' % (pointID, viewID))

        tmp0 = hmMeshCache.call_cache_hm_house_vert_world_0(
            houseID0=houseID, verbose=True)
        vertWorld0, face0 = tmp0['vertWorld0'], tmp0['face0']

        winSize = resolution0
        # fScale = 1. / math.tan(fovRadCurrent[k] / 2.)
        fScale = fScale0
        focalLength = fScale * winSize / 2.
        assert winSize == 512
        pyrenderManager.clear()
        pyrenderManager.add_plain_mesh(vertWorld0, face0)
        pyrenderManager.add_camera(
            np.array([focalLength, focalLength, winSize, winSize], dtype=np.float32),
            np.linalg.inv(ELU02cam0(np.concatenate([
                EWorld0, LWorld0, UWorld0,
            ], 0)))
        )
        tmp = pyrenderManager.render()
        ratio = omnidata_depth / tmp[1]
        print('The ratio is Min: %.6f, Max: %.6f, Mean: %.6f' %
              (ratio[np.isfinite(ratio)].min(),
               ratio[np.isfinite(ratio)].max(),
               ratio[np.isfinite(ratio)].mean(),
               ))
        summary0['omnidata_renderZBuffer_%d_%d' % (pointID, viewID)] = \
            to_heatmap(tmp[1], vmin=0.5, vmax=6, cmap='inferno')
        txt0.append('Point %d View %d renderZBuffer min %.3f max %.3f mean %.3f'
                    % (pointID, viewID, ratio[np.isfinite(ratio)].min(),
                       ratio[np.isfinite(ratio)].max(),
                       ratio[np.isfinite(ratio)].mean()))
        brInds.append(len(summary0))

        htmlStepper.step2(
            summary0, txt0, brInds,
            headerMessage='hmRenderOm %s (flagSplit: %d)' % (j, flagSplit[j]),
            subMessage='hm %05d-%s pointID %d viewID %d' %
                       (houseID, houseShaList_house[houseID], pointID, viewID)
        )


if __name__ == '__main__':
    main()
