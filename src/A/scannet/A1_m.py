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
import pickle
import numpy as np
from codes_py.np_ext.mat_io_v1 import pSaveMat
import pickle


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'

    scanIDList_trainval = sorted(os.listdir(projRoot + 'remote_fastdata/scannet/scans/'))
    scanIDList_test = sorted(os.listdir(projRoot + 'remote_fastdata/scannet/scans_test/'))
    m_trainval = len(scanIDList_trainval)
    m_test = len(scanIDList_test)

    scanIDList = scanIDList_trainval + scanIDList_test
    m = m_trainval + m_test
    ifTest = np.arange(m) >= m_trainval

    fileList = []
    for scanID in scanIDList_trainval:
        fileList.append('scans/' + scanID + '/')
    for scanID in scanIDList_test:
        fileList.append('scans_test/' + scanID + '/')

    scanIDDict = {scanIDList[j]: j for j in range(m)}

    pSaveMat(projRoot + 'v/A/%s/A1_m.mat' % dataset, {
        'm': m,
        'scanIDList': scanIDList,
        'ifTest': ifTest,
        'fileList': fileList,
    })
    with open(projRoot + 'v/A/%s/A1_m.pkl' % dataset, 'wb') as f:
        pickle.dump({
            'm': m,
            'scanIDList': scanIDList,
            'ifTest': ifTest,
            'fileList': fileList,
            'scanIDDict': scanIDDict,
        }, f)


if __name__ == '__main__':
    main()
