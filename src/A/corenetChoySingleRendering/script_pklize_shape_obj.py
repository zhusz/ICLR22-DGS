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
from codes_py.toolbox_3D.mesh_io_v1 import load_obj_np
import pickle
import numpy as np


def main():
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
    A1 = udl('pkl_A1b_', 'corenetChoySingleRendering')
    catIDList = A1['catIDList']
    shaIDList = A1['shaIDList']
    catShaIDList = [x[0] + '|' + x[1] for x in zip(catIDList, shaIDList)]

    d = {}
    for j, catShaID in enumerate(catShaIDList):
        d[catShaID] = j

    for i, catShaID in enumerate(d.keys()):
        print('Processing %d / %d' % (i, len(d)))
        assert catShaID[8] == '|'

        catID = catShaID[:8]
        shaID = catShaID[9:]

        vertObj0, face0 = load_obj_np(
            projRoot + 'remote_fastdata/shapenetcore_v2/%s/%s/models/model_normalized.obj' %
                       (catID, shaID),
        )
        vertObj0 = vertObj0.astype(np.float32)
        face0 = face0.astype(np.int32)

        # just for the purpose of faster runtime loading during training.
        with open(projRoot + 'v/R/shapenetv2/%s_%s.pkl' % (catID, shaID), 'wb') as f:
            pickle.dump({
                'vertObj0': vertObj0,
                'face0': face0,
            }, f)


if __name__ == '__main__':
    main()
