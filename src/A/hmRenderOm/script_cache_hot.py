# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
sys.path.append(projRoot + 'src/B/')
import numpy as np
import time
from Bpredwood2.testDataEntry.hmRenderOmDataset import HmRenderOmDataset
from Bpredwood2.testDataEntry.testDataEntryPool import getTestDataEntryDict
from skimage.io import imread
from UDLv3 import udl


def main():
    hm3d_zbuffer_raw_root = projRoot + 'remote_fastdata/omnidata/om/depth_zbuffer/hm3d/'
    hm3d_rgb_raw_root = projRoot + 'remote_fastdata/omnidata/om/rgb/hm3d/'

    assert os.path.isdir(hm3d_zbuffer_raw_root), hm3d_zbuffer_raw_root
    assert os.path.isdir(hm3d_rgb_raw_root), hm3d_rgb_raw_root

    dataset = os.path.realpath(__file__).split('/')[-2]

    A1 = udl('pkl_A1_', dataset)
    m = A1['m']
    houseIDList = A1['houseIDList']
    pointIDList = A1['pointIDList']
    viewIDList = A1['viewIDList']
    dataset_house = A1['houseDataset']

    A1_house = udl('pkl_A1_', dataset_house)
    houseShaList_house = A1_house['houseShaList']

    j1 = int(os.environ['J1'])
    j2 = int(os.environ['J2'])
    assert 0 <= j1 < j2
    indChosen = list(range(j1, j2))

    datasetConf = {
        'dataset': 'hmRenderOm', 'trainingSplit': 'train', 'batchSize': 32,
        'class': 'HmRenderOmDataset',

        # Sampling
        'samplingMethodList': ['fullsurface', 'nearsurface', 'selfnonsurface'],

        'numSamplingFinalFullsurface': 2048 * 16,

        'numSamplingFinalNearsurface': 2048,
        'numSamplingSufficientNearsurface': 2048 * 8,
        'nearsurfaceDeltaRange': 0.016 * 4,

        'numSamplingFinalSelfnonsurface': 512,
        'numSamplingSufficientSelfnonsurface': 512 * 8,
        'selfnonsurfaceDeltaRange': 0.008 * 4,

        'winWidth': 256,
        'winHeight': 256,
        # These params are no longer from only the datasetConf (because they are now likely to be variant on different samples)
        # although in this S, they are still fixed.
        # But you can no longer assume this anymore.
        # 'focalLengthWidth': 230.4, 
        # 'focalLengthHeight': 230.4,
        # 'fScaleWidth': 1.8,
        # 'fScaleHeight': 1.8,

        'zNear': 1.e-6,

        'ifNeedPixelAugmentation': 0,
        'ifNeedCroppingAugmentation': 1,
        'fScaleWidthDesiredFixed': 1.8,
        'fScaleHeightDesiredFixed': 1.8,
        'ifNeedMirrorAugmentation': 0,

        'ifDummy': False,
    }

    datasetObj = HmRenderOmDataset(datasetConf, projRoot=projRoot, datasetSplit='train')   
    
    testDataEntryDict = getTestDataEntryDict(wishedTestDataNickName=['hmFsOfficialValSplit10'])\
        ['hmFsOfficialValSplit10']

    flagSplit = np.zeros((m, ), dtype=np.int32)
    flagSplit[datasetObj.flagSplit == 1] = 1
    flagSplit[testDataEntryDict['indChosen']] = 2

    for j in indChosen:
        if j % 100 == 0:
            if_to_print = True

        if flagSplit[j] == 0:
            continue

        t = time.time()

        houseID0 = int(houseIDList[j])
        houseSha0 = houseShaList_house[houseID0]
        pointID0 = int(pointIDList[j])
        viewID0 = int(viewIDList[j])
        
        rgb = hm3d_rgb_raw_root + '%05d-%s/point_%d_view_%d_domain_rgb.png' % \
            (houseID0, houseSha0, pointID0, viewID0)
        tmp0 = imread(rgb)
        depth_zbuffer = hm3d_zbuffer_raw_root + '%05d-%s/point_%d_view_%d_domain_depth_zbuffer.png' % \
            (houseID0, houseSha0, pointID0, viewID0)
        tmp1 = imread(depth_zbuffer)
        if if_to_print:
            print('script_cache_hot for %s: %8d (J1 = %8d, J2 = %8d), Elapsed Time: %.3f seconds' % 
                (dataset, j, j1, j2, time.time() - t))
            if_to_print = False


if __name__ == '__main__':
    main()

