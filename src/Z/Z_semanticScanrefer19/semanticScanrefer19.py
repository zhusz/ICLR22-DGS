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
import pickle
import numpy as np


def main():
    type2class = {
        'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
        'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
        'refridgerator': 12, 'shower curtain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16,
        'others': 17, 'structures': 18}
    for k in type2class.keys():
        type2class[k] += 1
    type2class['air'] = 0

    scanrefer19Dict = type2class
    scanrefer19RetrieveList = {scanrefer19Dict[k]: k for k in scanrefer19Dict.keys()}
    scanrefer19List = []
    for c in range(len(scanrefer19Dict.keys())):
        scanrefer19List.append(scanrefer19RetrieveList[c])

    # map to nyu40
    nyu40 = udl('_Z_semanticNYU')
    nyu40RetrieveList = nyu40['nyu40RetrieveList']
    nyu40List = nyu40['nyu40List']
    for scanrefer19ClassName in scanrefer19RetrieveList.values():
        if scanrefer19ClassName not in ['others', 'structures']:
            assert scanrefer19ClassName in nyu40List, scanrefer19ClassName
    nyu40ToScanrefer19 = -np.ones((41, ), dtype=np.int32)
    for c in range(41):
        nyu40ClassName = nyu40RetrieveList[c]
        if nyu40ClassName in ['wall', 'floor', 'ceiling']:
            nyu40ToScanrefer19[c] = scanrefer19Dict['structures']
        elif nyu40ClassName in list(scanrefer19Dict.keys()):
            nyu40ToScanrefer19[c] = scanrefer19Dict[nyu40ClassName]
        else:
            nyu40ToScanrefer19[c] = scanrefer19Dict['others']

    with open(projRoot + 'v/Z/semanticScanrefer/semanticScanrefer.pkl', 'wb') as f:
        pickle.dump({
            'scanrefer19List': scanrefer19List,
            'scanrefer19Dict': scanrefer19Dict,
            'scanrefer19RetrieveList': scanrefer19RetrieveList,
            'nyu40ToScanrefer19': nyu40ToScanrefer19,
        }, f)


if __name__ == '__main__':
    main()
