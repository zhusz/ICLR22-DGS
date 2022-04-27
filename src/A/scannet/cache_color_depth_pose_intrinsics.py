# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
import subprocess
from SensorData import SensorData
from codes_py.np_ext.mat_io_v1 import pLoadMat, pSaveMat
from multiprocessing import Pool
import math


def main(d):
    rank, numMpProcess = d

    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'

    # A1 = udl('_A1_', dataset)
    A1 = pLoadMat(projRoot + 'v/A/scannet/A1_m.mat')
    m = A1['m']
    scanIDList = A1['scanIDList']
    fileList = A1['fileList']

    j1 = int(math.floor(float(rank) / max(numMpProcess, 1) * m))
    j2 = int(math.floor(float(rank + 1) / max(numMpProcess, 1) * m))
    indChosen = list(range(j1, j2))

    count = 0
    for j in indChosen:
        cacheRoot = projRoot + 'remote_fastdata/scannet/scannet_cache/%s' % fileList[j]
        finishTagFile = cacheRoot + 'finishTag.mat'

        if os.path.isfile(finishTagFile):
            mat = pLoadMat(finishTagFile)
            if mat['tag'] == 1:
                print('Processing cahce_color_depth_pose_intrinsics for %s-%d'
                      '(j1 = %d, j2 = %d, count = %d, tot = %d)' %
                      (dataset, j, j1, j2, count, len(indChosen)))
                continue

        print('Processing cahce_color_depth_pose_intrinsics for %s-%d'
              '(j1 = %d, j2 = %d, count = %d, tot = %d, rank = %d)' %
              (dataset, j, j1, j2, count, len(indChosen), rank))
        sd = SensorData(projRoot + 'remote_fastdata/scannet/%s%s.sens' %
                        (fileList[j], scanIDList[j]))
        sd.export_depth_images(cacheRoot + 'depth')
        sd.export_color_images(cacheRoot + 'color')
        sd.export_poses(cacheRoot + 'pose')
        sd.export_intrinsics(cacheRoot + 'intrinsic')
        pSaveMat(finishTagFile, {'tag': 1})

        count += 1

    return None


if __name__ == '__main__':
    numMpProcess = int(os.environ['MP'])
    if numMpProcess <= 0:
        main((0, 0))
    else:
        p = Pool(numMpProcess)
        print('Entering Pool')
        p.map(main, [(rank, numMpProcess) for rank in range(numMpProcess)])
