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


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))

    A1 = udl('pkl_A1_', dataset)
    m = A1['m']
    sceneTagList = A1['sceneTagList']

    A1_scannet = udl('pkl_A1_', 'scannet')
    scanIDList_scannet = A1_scannet['scanIDList']
    m_scannet = A1_scannet['m']
    flagSplit_scannet = udl('mat_A01_flagSplit', 'scannet')

    flagSplit = np.zeros((m, ), dtype=np.int32)
    for j in range(m):
        print('Processing A01_ind for %s: %d / %d' % (dataset, j, m))
        tmp = [j_scannet for j_scannet in range(m_scannet)
               if sceneTagList[j] == scanIDList_scannet[j_scannet]]
        assert len(tmp) == 1, (sceneTagList[j], tmp)
        j_scannet = int(tmp[0])
        flagSplit[j] = flagSplit_scannet[j_scannet]

    pSaveMat(projRoot + 'v/A/%s/A01_ind.mat' % dataset, {
        'flagSplit': flagSplit,
    })


if __name__ == '__main__':
    main()

