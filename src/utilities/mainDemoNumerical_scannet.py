# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (tfconda)
import torch
import os
import sys
import numpy as np
import copy
from configs_registration import getConfigGlobal
from codes_py.py_ext.misc_v1 import mkdir_full
from codes_py.toolbox_framework.framework_util_v2 import castAnything
from codes_py.np_ext.mat_io_v1 import pLoadMat, pSaveMat


bt = lambda x: x[0].upper() + x[1:]


def mainDemoNumerical(rank, numMpProcess):
    fullPathSplit = os.path.dirname(os.path.realpath(__file__)).split('/')
    P = fullPathSplit[-1]
    D = os.environ['D']
    S = os.environ['S']
    R = os.environ['R']
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'

    getConfigFunc = getConfigGlobal(P, D, S, R)['getConfigFunc']
    config = getConfigFunc(P, D, S, R)
    if 'I' in list(os.environ.keys()):
        iter_label = int(os.environ['I'][1:])
    else:
        iter_label = None
    assert iter_label is not None

    # net
    DTrainer = getConfigGlobal(P, D, S, R)['exportedClasses']['DTrainer']
    trainer = DTrainer(config, rank=rank, numMpProcess=numMpProcess, ifDumpLogging=False)

    # batch_np
    batch_np = pLoadMat(trainer.logDir + 'dump/train_iter_%d.mat' % iter_label)

    # model train mode
    trainer.initializeAll(iter_label=iter_label, hook_type='stat', ifLoadToCuda=True)
    for k in trainer.models.keys():
        if k != 'meta' and k not in trainer.models['meta']['nonLearningModelNameList']:
            trainer.models[k].train()
    batch_thcpu = castAnything(batch_np, 'np2thcpu')
    batch_thcpu = {k: batch_thcpu[k].contiguous() for k in batch_thcpu.keys()
                   if type(batch_thcpu[k]) is torch.Tensor}
    batch_thgpu = castAnything(batch_np, 'np2thgpu')
    batch_thgpu = {k: batch_thgpu[k].contiguous() for k in batch_thgpu.keys()
                   if type(batch_thgpu[k]) is torch.Tensor}
    batch_thgpu = trainer.batchPreprocessingTHGPU(
        batch_thcpu, batch_thgpu, datasets=trainer.datasets, datasetMetaConf=config.datasetMetaConf,
        iterCount=int(iter_label),
    )
    batch_thgpu = trainer.forwardBackwardUpdate(
        batch_thgpu, ifTrain=True, iterCount=iter_label, datasetMetaConf=config.datasetMetaConf,
        ifRequiresGrad=False, ifAllowBackward=False,
    )
    d = {}
    for k in trainer.models.keys():
        if k != 'meta' and k not in trainer.models['meta']['nonLearningModelNameList']:
            d.update({
                'o%s_trainMode_%d' % (k, trainer.resumeIter):
                    trainer.hookModels[k].hooker_output_list,
                'p%s_trainMode_%d' % (k, trainer.resumeIter):
                    trainer.hookModels[k].hooker_params_list,
            })
    pSaveMat(trainer.logDir + 'numerical/trainModeFlow_iter%d.mat' % trainer.resumeIter, d)

    # model eval mode
    trainer.initializeAll(iter_label=iter_label, hook_type='stat', ifLoadToCuda=True)
    for k in trainer.models.keys():
        if k != 'meta' and k not in trainer.models['meta']['nonLearningModelNameList']:
            trainer.models[k].eval()
    batch_thcpu = castAnything(batch_np, 'np2thcpu')
    batch_thcpu = {k: batch_thcpu[k].contiguous() for k in batch_thcpu.keys()
                   if type(batch_thcpu[k]) is torch.Tensor}
    batch_thgpu = castAnything(batch_np, 'np2thgpu')
    batch_thgpu = {k: batch_thgpu[k].contiguous() for k in batch_thgpu.keys()
                   if type(batch_thgpu[k]) is torch.Tensor}
    batch_thgpu = trainer.batchPreprocessingTHGPU(
        batch_thcpu, batch_thgpu, datasets=trainer.datasets, datasetMetaConf=config.datasetMetaConf,
        iterCount=int(iter_label)
    )
    batch_thgpu = trainer.forwardBackwardUpdate(
        batch_thgpu, ifTrain=False, iterCount=iter_label, datasetMetaConf=config.datasetMetaConf,
        ifRequiresGrad=False, ifAllowBackward=False,
    )
    d = {}
    for k in trainer.models.keys():
        if k != 'meta' and k not in trainer.models['meta']['nonLearningModelNameList']:
            d.update({
                'o%s_valMode_%d' % (k, trainer.resumeIter):
                    trainer.hookModels[k].hooker_output_list,
                'p%s_valMode_%d' % (k, trainer.resumeIter):
                    trainer.hookModels[k].hooker_params_list,
            })
    pSaveMat(trainer.logDir + 'numerical/valModeFlow_iter%d.mat' % trainer.resumeIter, d)


if __name__ == '__main__':
    mainDemoNumerical(rank=0, numMpProcess=0)
