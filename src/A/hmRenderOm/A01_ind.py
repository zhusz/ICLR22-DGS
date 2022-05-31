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
from UDLv3 import udl
import numpy as np
from codes_py.np_ext.mat_io_v1 import pSaveMat
from skimage.io import imread
from multiprocessing import Pool
import time


def f(input):
    houseID0, houseSha, pointID0, viewID0, flagSplit_house = input
    hm3d_zbuffer_raw_root = projRoot + 'remote_fastdata/omnidata/om/depth_zbuffer/hm3d/'
    depth0 = imread(
        hm3d_zbuffer_raw_root + '%05d-%s/point_%d_view_%d_domain_depth_zbuffer.png' %
        (houseID0, houseSha, pointID0, viewID0)).astype(np.float32) / 512.
    depth0[np.isfinite(depth0) == 0] = np.nan
    depth0[depth0 < 0.05] = np.nan
    depth0[depth0 > 10] = np.nan

    finite_depth = depth0[np.isfinite(depth0)]
    if houseID0 != 887 and finite_depth.shape[0] > 0:  # houseID == [480, 887] should be excluded
        flag1 = np.mean(finite_depth) >= 1.0
        flag2 = np.min(finite_depth) >= 0.2
        flag3 = (float(finite_depth.shape[0]) / float(np.prod(depth0.shape))) >= 0.7
        flag4 = True
    else:
        flag1, flag2, flag3, flag4 = False, False, False, False
    if flag1 and flag2 and flag3 and flag4:
        flagSplit = flagSplit_house
    else:
        flagSplit = 0
    return (flagSplit, flag1, flag2, flag3)


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    A1 = udl('pkl_A1_', dataset)
    m = A1['m']
    houseIDList = A1['houseIDList']
    pointIDList = A1['pointIDList']
    viewIDList = A1['viewIDList']
    dataset_house = A1['houseDataset']
    assert dataset_house == 'hm'
    A1_house = udl('pkl_A1_', dataset_house)
    houseShaList_house = A1_house['houseShaList']
    flagSplit_house = udl('_A01_flagSplit', dataset_house)

    # flag123 = np.zeros((m, 3), dtype=bool)
    # flagSplit = np.zeros((m,), dtype=np.int32)

    num_threads = 32
    m_to_process = m
    print('Starting...')
    t = time.time()
    feed_in = [
        (
            int(houseIDList[j]),
            houseShaList_house[int(houseIDList[j])],
            int(pointIDList[j]),
            int(viewIDList[j]),
            int(flagSplit_house[int(houseIDList[j])]),
        )
        for j in range(m_to_process)
    ]
    with Pool(num_threads) as p:
        out = p.map(f, feed_in)
    print('Elapsed Time is %.3f seconds.' % (time.time() - t))
    tmp = np.array(out, dtype=np.int32)
    flagSplit = tmp[:, 0].astype(np.int32)
    flag123 = tmp[:, 1:4].astype(bool)

    '''
    for j in range(m):
        if j % 100 == 0:
            print('Processing A01_ind for %s: %d / %d' % (dataset, j, m))

        houseID0 = int(houseIDList[j])
        houseSha = houseShaList_house[houseID0]
        pointID0 = int(pointIDList[j])
        viewID0 = int(viewIDList[j])

        depth0 = imread(
            hm3d_zbuffer_raw_root + '%05d-%s/point_%d_view_%d_domain_depth_zbuffer.png' %
            (houseID0, houseSha, pointID0, viewID0)).astype(np.float32) / 512.
        depth0[np.isfinite(depth0) == 0] = np.nan
        depth0[depth0 < 0.05] = np.nan
        depth0[depth0 > 10] = np.nan

        finite_depth = depth0[np.isfinite(depth0)]
        if houseID0 != 887 and finite_depth.shape[0] > 0:
            flag1 = np.mean(finite_depth) >= 1.0
            flag2 = np.min(finite_depth) >= 0.2
            flag3 = (float(finite_depth.shape[0]) / float(np.prod(depth0.shape))) >= 0.7
            flag4 = True
        else:
            flag1, flag2, flag3, flag4 = False, False, False, False

        flag123[j, 0], flag123[j, 1], flag123[j, 2] = flag1, flag2, flag3
        if flag1 and flag2 and flag3 and flag4:
            flagSplit[j] = flagSplit_house[houseID0]
        else:
            flagSplit[j] = 0
    '''

    pSaveMat(projRoot + 'v/A/%s/A01_ind.mat' % dataset, {
        'flagSplit': flagSplit,
        'flag123': flag123,
    })


if __name__ == '__main__':
    main()
