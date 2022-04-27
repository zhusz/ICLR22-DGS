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
import torch.multiprocessing as mp
from configs_registration import getConfigGlobal


def main():
    fullPathSplit = os.path.dirname(os.path.realpath(__file__)).split('/')
    P = fullPathSplit[-1]
    D = os.environ['D']
    S = os.environ['S']
    R = os.environ['R']
    getConfigFunc = getConfigGlobal(P, D, S, R)['getConfigFunc']
    config = getConfigFunc(P, D, S, R)

    DTrainer = getConfigGlobal(P, D, S, R)['exportedClasses']['DTrainer']

    trainer = DTrainer(config, ifDumpLogging=True)
    print(trainer.config)

    if 'I' in list(os.environ.keys()):
        iter_label = int(os.environ['I'][1:])
    else:
        iter_label = None

    trainer.initializeAll(iter_label=iter_label, hook_type=None, ifLoadToCuda=True)
    trainer.train()


if __name__ == '__main__':
    main()
