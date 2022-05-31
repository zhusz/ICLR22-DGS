# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (tfconda)
import os
from codes_py.py_ext.misc_v1 import mkdir_full
from skimage.io import imread, imsave
import numpy as np


def main():
    raw_dir = '/home/zhusz/omnidata/downloads/uncompressed/'
    d_dir = raw_dir + 'depth_zbuffer/taskonomy/'
    m_dir = raw_dir + 'mask_valid/taskonomy/'
    sceneNameList = sorted(os.listdir(d_dir))

    j1 = int(os.environ['J1'])
    j2 = int(os.environ['J2'])
    assert 0 <= j1 < j2

    for i, scene_name in enumerate(sceneNameList):
        print('Populating %d / %d' % (i, len(sceneNameList)))

        if i < j1 or i >= j2:
            continue

        mkdir_full(m_dir + scene_name + '/')
        fns = sorted(os.listdir(d_dir + scene_name + '/'))

        if len(os.listdir(m_dir + scene_name)) == \
                len(os.listdir(d_dir + scene_name)):
            continue
        # else:
        #     print('Start from here.')
        #     import ipdb
        #     ipdb.set_trace()
        #     print(1 + 1)

        for fn in fns:
            d0 = imread(d_dir + scene_name + '/' + fn)
            assert 'uint16' in str(d0.dtype)
            assert len(d0.shape) == 2
            assert fn.endswith('_depth_zbuffer.png')

            m0 = 255 * np.ones((d0.shape[0], d0.shape[1]), dtype=np.uint8)
            m0[(d0 == 65535) | (d0 == 0)] = 0

            imsave(m_dir + scene_name + '/' + fn[:-18] + '_mask_valid.png', m0)


if __name__ == '__main__':
    main()

