# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (tfconda)
import os
import numpy as np
from codes_py.np_ext.mat_io_v1 import pSaveMat
import pickle


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
    dataset_raw_root = projRoot + 'remote_fastdata/hm/install/'

    fns = sorted(os.listdir(dataset_raw_root))
    assert len(fns) == 900
    houseShaList = []
    mtlShaList = []
    for j, fn in enumerate(fns):
        assert fn.startswith('%05d-' % j)
        assert len(fn) == 17
        houseShaList.append(fn[6:])
        fs = [f for f in os.listdir(dataset_raw_root + '%05d-%s' % (j, houseShaList[j]))
              if f.endswith('.mtl')]
        assert len(fs) == 1
        assert len(fs[0]) == 36
        mtlShaList.append(fs[0][:-4])
    m = 900
    flagSplit = np.zeros((m, ), dtype=np.int32)
    flagSplit[:800] = 1
    flagSplit[800:] = 2

    pSaveMat(projRoot + 'remote_slowdata/pp/v/A/hm/A01_ind.mat', {
        'flagSplit': flagSplit,
    })
    with open(projRoot + 'remote_slowdata/pp/v/A/hm/A1_m.pkl', 'wb') as f:
        pickle.dump({
            'm': m,
            'houseShaList': houseShaList,
            'mtlShaList': mtlShaList,
        }, f)

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()
