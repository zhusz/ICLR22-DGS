# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
sys.path.append(projRoot + 'src/versions/')
from drawing import drawing


def main():
    fn = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]

    curveTupList = [
        ('PredwoodA32D0dgsShybridR0', None, None, 'blue', '-'),
    ]

    for key in [
        'lossDepthRegL1',
        'lossNearsurfaceClf',
        'statTrainFinalBenFitF1',
        'statValFinalBenFitF1',
    ]:
        curveTupListUpdated = [
            (x[0], key, x[2], x[3], x[4]) for x in curveTupList
        ]
        drawing(curveTupListUpdated, fn + key, ylim=(0.0, 0.7))


if __name__ == '__main__':
    main()
