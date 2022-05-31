# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (tfconda)
import os
import subprocess
from skimage.io import imread
import numpy as np
import pickle


def main():
    commandList = [
        'find',
        '/home/zhusz/omnidata/d2/uncompressed/mask_valid/',
        '-name',
        '*.png',
    ]
    # result = subprocess.check_call(commandList)
    output = subprocess.check_output(commandList)
    fns = sorted(str(output)[2:-3].split('\\n'))

    # fns = [fn for fn in fns if 'blended_mvg' not in fn]
    fns = fns[::500]

    collector = {}
    for i, fn in enumerate(fns):
        if i % 100 == 0:
            print('Checking in progress: %d / %d' % (i, len(fns)))
        img0 = imread(fn)
        assert 'uint8' in str(img0.dtype)
        assert len(np.unique(img0)) <= 2
        assert int(img0.min()) in [0, 255]
        assert int(img0.max()) in [0, 255]
        assert fn.endswith('_mask_valid.png')

        if img0.min() < 255:
            r = fn.find('mask_valid')

            dfn = fn[:r] + 'depth_zbuffer' + fn[r + 10:-15] + '_depth_zbuffer.png'
            d0 = imread(dfn)
            assert 'uint16' in str(d0.dtype)
            assert d0.shape == img0.shape
            tmp0 = d0[img0 == 0]

            dfn = fn[:r] + 'depth_euclidean' + fn[r + 10:-15] + '_depth_euclidean.png'
            d1 = imread(dfn)
            assert 'uint16' in str(d1.dtype)
            assert d1.shape == img0.shape
            tmp1 = d1[img0 == 0]

            if not ((tmp0 == 65535) | (tmp1 == 65536)).all():
                print('Problem in here: %d, %s' % (i, fn))
                collector[i] = {
                    'd0': d0,
                    'd1': d1,
                    'img0': img0,
                    'invalid_0': tmp0,
                    'invalid_1': tmp1,
                    'fn': fn,
                }

        '''
        if not img0.min() == 255 and fn not in waiveFnList:
            print('Problem occurs. Please Check! i: %d, fn: %s' % (i, fn))
            import ipdb
            ipdb.set_trace()
            print(1 + 1)
        '''

    # with open('/home/zhusz/local/r/pp/cache/dataset/omnidata/'
    #           'taskonomy_check_mask_valid.pkl', 'wb') as f:
    #     pickle.dump(collector, f)

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()
