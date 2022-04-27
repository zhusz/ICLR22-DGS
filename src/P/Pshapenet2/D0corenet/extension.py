# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


def returnExportedClasses(wishedClassNameList):  # To avoid import unnecessary class of different envs

    _testSuiteD = os.path.basename(os.path.dirname(__file__))

    exportedClasses = {}

    if wishedClassNameList is None or 'DCorenetChoySingleRenderingDataset' in wishedClassNameList:
        from ..dataset import PCorenetChoySingleRenderingDataset
        class DCorenetChoySingleRenderingDataset(PCorenetChoySingleRenderingDataset):
            pass
        exportedClasses['DCorenetChoySingleRenderingDataset'] = DCorenetChoySingleRenderingDataset

    if wishedClassNameList is None or 'DTrainer' in wishedClassNameList:
        from ..trainer import Trainer
        class DTrainer(Trainer):
            pass
        exportedClasses['DTrainer'] = DTrainer

    return exportedClasses

