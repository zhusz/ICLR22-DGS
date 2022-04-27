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
from UDLv3 import udl
import numpy as np
from codes_py.np_ext.mat_io_v1 import pSaveMat


def wrong():
    print('Problem occurred. Please check!')
    import ipdb
    ipdb.set_trace()
    print(1 + 1)


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'

    lists = {}
    for split in ['train', 'val', 'test']:
        with open(projRoot + 'remote_fastdata/cache/scannet/all_tsdf_9/splits/scannetv2_%s.txt' %
                  split, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lists[split] = lines

    A1 = udl('_A1_', dataset)
    m = A1['m']
    scanIDList = A1['scanIDList']
    ifTest = A1['ifTest']

    flagSplit = -np.ones(m, dtype=np.int32)
    count = 0
    for split, splitID in [('train', 1), ('val', 2), ('test', 3)]:
        for line in lists[split]:
            if count % 100 == 0:
                print('Processing A01_ind flagSplit for %s: (count = %d, tot = %d)' %
                      (dataset, count, m))
            j = scanIDList.index(line)
            if not (0 <= j < m):
                wrong()
            if ((split in ['train', 'val']) and (ifTest[j] != 0)) or \
                ((split in ['test']) and (ifTest[j] != 1)):
                wrong()
            flagSplit[j] = splitID

            count += 1

    assert flagSplit.min() == 1
    assert flagSplit.max() == 3

    pSaveMat(projRoot + 'v/A/%s/A01_ind.mat' % dataset, {
        'flagSplit': flagSplit,
    })


if __name__ == '__main__':
    main()
