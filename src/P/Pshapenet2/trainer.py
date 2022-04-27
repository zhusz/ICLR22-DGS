# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Basic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import grid_sample
import torchvision
import torch.optim as optim
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler
from collections import OrderedDict
import logging
import os
import time
import math
from copy import deepcopy
import copy
import numpy as np
import sys
import io
import re
import random
from socket import gethostname
import cc3d

# toolbox
from UDLv3 import udl
from codes_py.toolbox_framework.framework_util_v4 import \
    checkPDSRLogDirNew, castAnything, probe_load_network, load_network, save_network, \
    bsv02bsv, mergeFromAnotherBsv0, bsv2bsv0, constructInitialBatchStepVis0, \
    batchExtractParticularDataset
from codes_py.toolbox_torch.hook_v1 import PyTorchForwardHook
from codes_py.np_ext.mat_io_v1 import pSaveMat
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoTHGPU, vertInfo2faceVertInfoNP
from codes_py.toolbox_3D.sdf_from_multiperson_v1 import sdfTHGPU
from pytorch3d.ops.knn import knn_points
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
from codes_py.toolbox_3D.rotations_v1 import camSys2CamPerspSysTHGPU
from codes_py.toolbox_3D.rotations_v1 import camSys2CamPerspSys0
from codes_py.toolbox_show_draw.html_v1 import HTMLStepperNoPrinting
from codes_py.np_ext.data_processing_utils_v1 import determinedArbitraryPermutedVector2correct
from codes_py.toolbox_3D.self_sampling_v1 import mesh_sampling_given_normal_np_simple
# from codes_py.toolbox_3D.pyrender_wrapper_v1 import PyrenderManager
from codes_py.toolbox_3D.dgs_wrapper_v1 import DGSLayer
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_3D.view_query_generation_v1 import gen_viewport_query_given_bound
from codes_py.toolbox_3D.representation_v1 import voxSdfSign2mesh_skmc, directDepthMap2PCNP
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly
from codes_py.corenet.geometry.voxelization import voxelize_mesh
from codes_py.corenet.cc import fill_voxels

# PDSR
from .dataset import PCorenetChoySingleRenderingDataset
from . import resnet50
from .decoder import Decoder
from . import file_system as fs

# benchmarking
from Bpshapenet2.testDataEntry.testDataEntryPool import getTestDataEntryDict
from Bpshapenet2.csvShapenetEntry.benchmarkingShapenetCorenet import \
    benchmarkingShapenetCorenetFunc
from Bpshapenet2.csvShapenetEntry.dumpHtmlForPrepickShapenet import addToSummary0Txt0BrInds

# inlines
bt = lambda s: s[0].upper() + s[1:]


class Trainer(object):
    @staticmethod
    def setup_for_distributed(is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print

    @staticmethod
    def get_trailing_number(s):
        m = re.search(r'\d+$', s)
        return int(m.group()) if m else 0

    def __init__(self, config, **kwargs):
        self.rank = kwargs['rank']
        self.numMpProcess = kwargs['numMpProcess']
        self.ifDumpLogging = kwargs['ifDumpLogging']
        self.trackedRank = 0 if self.numMpProcess <= 0 else self.numMpProcess - 1
        self.projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
        if not config.R.startswith('Rtmp'):
            assert not config.debug

        # set device.
        self.cudaDeviceForAll = 'cuda:%d' % self.rank
        torch.cuda.set_device(self.cudaDeviceForAll)

        if self.numMpProcess <= 0:  # single threaded
            assert self.rank == 0
            self.logDir = checkPDSRLogDirNew(
                config,
                projRoot=self.projRoot,
                ifInitial=config.R.startswith('Rtmp'))
        else:  # DDP
            dist.init_process_group(
                backend="nccl", rank=self.rank, world_size=self.numMpProcess)
            dist.barrier()
            self.setup_for_distributed(self.rank == self.trackedRank)
            dist.barrier()
            if self.rank == self.trackedRank:
                self.logDir = checkPDSRLogDirNew(
                    config,
                    projRoot=self.projRoot,
                    ifInitial=config.R.startswith('Rtmp'))
            else:
                self.logDir = self.projRoot + 'v/P/%s/%s/%s/%s/' % (
                    config.P, config.D, config.S, config.R
                )
            dist.barrier()

        # logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        if self.rank == self.trackedRank or self.numMpProcess == 0:
            self.logger.addHandler(ch)
        fh = logging.FileHandler(self.logDir + 'trainingLog.txt')
        fh.setLevel(logging.INFO)
        if (self.rank == self.trackedRank or self.numMpProcess == 0) and self.ifDumpLogging:
            self.logger.addHandler(fh)
        self.logger.propagate = False
        self.logger.info('====== Starting training: Host %s GPU %s CPU %d Rank %d numMpProcess %d '
                         '%s %s %s %s ======' %
                         (gethostname(),
                          os.environ['CUDA_VISIBLE_DEVICES'], os.getpid(),
                          self.rank, self.numMpProcess,
                          config.P, config.D, config.S, config.R))

        # random seeding  # the ending integer
        self.seed = self.get_trailing_number(config.R) + self.rank
        self.logger.info('[Random Seed] seed = %d, R = %s, trailing number = %d, rank = %d' %
                      (self.seed, config.R,
                       self.get_trailing_number(config.R), self.rank))
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # cuda backend
        torch.backends.cudnn.benchmark = True

        # meta params (not influencing model training)
        self.printFreq = 10
        self.minSnapshotFreq = 2500
        self.samplingVerbose = False
        self.monitorTrainFreq = 1000
        self.monitorValFreq = 100000
        self.monitorDumpFreq = 200000
        self.monitorMode = 0  # which iteration is the first to step the monitors

        # runnerComplementTheConfig
        self.config = self._runnerComplementTheConfig(config, numMpProcess=self.numMpProcess)

    @staticmethod
    def _runnerComplementTheConfig(config, **kwargs):
        # This function cannot be called during demo time. Demo does not allow multiple GPU.
        numMpProcess = kwargs['numMpProcess']
        datasetConfDict = config.datasetConfDict
        for datasetConf in datasetConfDict.values():
            if numMpProcess > 0:
                keys = list(datasetConf.keys())
                for k in keys:
                    if k.startswith('batchSize'):
                        datasetConf[k + 'PerProcess'] = int(datasetConf[k] / numMpProcess)
            else:
                keys = list(datasetConf.keys())
                for k in keys:
                    if k.startswith('batchSize'):
                        datasetConf[k + 'PerProcess'] = datasetConf[k]

        return config

    def metaDataLoading(self, **kwargs):
        config = self.config
        self.logger.info('[Trainer] MetaDataLoading')
        projRoot = self.projRoot

        datasetConfDict = config.datasetConfDict

        datasetObjDict = OrderedDict([])
        dataLoaderDict = OrderedDict([])
        dataLoaderIterDict = OrderedDict([])

        for datasetConf in datasetConfDict.values():
            if datasetConf['class'] == 'CorenetChoySingleRenderingDataset':
                datasetObj = PCorenetChoySingleRenderingDataset(
                    datasetConf,
                    datasetSplit=datasetConf['trainingSplit'],
                    projRoot=projRoot,
                    datasetMode='trainFast',
                )
                batchSizePerProcess = datasetConf['batchSizePerProcess']
                num_workers = int(math.ceil(batchSizePerProcess / 8.))
                dataLoader = torch.utils.data.DataLoader(
                    datasetObj, batch_size=batchSizePerProcess, num_workers=num_workers,
                    shuffle=True, drop_last=True,
                )
                datasetObjDict[datasetConf['dataset']] = datasetObj
                dataLoaderDict[datasetConf['dataset']] = dataLoader
                dataLoaderIterDict[datasetConf['dataset']] = iter(dataLoader)
            else:
                raise NotImplementedError('Unknown dataset class: %s' % datasetConf['class'])
        self.datasetObjDict = datasetObjDict
        self.dataLoaderDict = dataLoaderDict
        self.dataLoaderIterDict = dataLoaderIterDict

    def _netConstruct(self, **kwargs):
        config = self.config
        self.logger.info('[Trainer] MetaModelLoading - _netConstruct')

        encoder = resnet50.ResNet50FeatureExtractor(
            config=config, cudaDevice=self.cudaDeviceForAll)
        encoder.load_state_dict(
            torch.load(
                io.BytesIO(fs.read_bytes(
                    self.projRoot + 'remote_syncdata/ICLR22-DGS/v_external_codes/corenet/data/keras_resnet50_imagenet.cpt'
                ))
            ), strict=False
        )

        networkTwo = Decoder(
            config=config, cudaDevice=self.cudaDeviceForAll)

        meta = {
            'nonLearningModelNameList': [],
        }
        self.models = dict()
        self.models['meta'] = meta
        self.models['encoder'] = encoder.to(self.cudaDeviceForAll)
        self.models['networkTwo'] = networkTwo.to(self.cudaDeviceForAll)

    def _netRandomInitialization(self, iter_label):
        # random initialization
        # (Different threads might result in different model weights. Only Rank 0 is used)
        self.logger.info('[Trainer] MetaModelLoading - _netRandomInitialization')

        if iter_label is not None:
            # So by definition these nan should be replaced with normal numbers later on.
            print('    NaN Init Robust Test... As iter_label is %s' % iter_label)
            for k_model, model in self.models.items():
                if k_model not in self.models['meta']['nonLearningModelNameList'] and \
                        k_model not in ['meta']:
                    for k, v in model.state_dict().items():
                        if 'num_batches_tracked' not in k:
                            # print('%s - %s' % (k_model, k))
                            v.fill_(torch.nan)
        pass

    def _netResumingInitialization(self, **kwargs):
        iter_label = kwargs['iter_label']
        self.logger.info('[Trainer] MetaModelLoading - _netResumingInitialization (iter_label = %s)' %
                         str(iter_label))
        if self.numMpProcess > 0:
            dist.barrier()
        if iter_label is None:
            resumeIter = probe_load_network(self.logDir)
        elif type(iter_label) is str:
            if iter_label == 'latest':
                resumeIter = 'latest'
            else:
                resumeIter = int(iter_label)
        else:
            assert type(iter_label) is int
            resumeIter = iter_label
        if self.rank == 0 and (resumeIter == 'latest' or resumeIter >= 0):  # model manager rank is always 0
            for k in self.models.keys():
                if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                    _, self.models[k] = load_network(
                        self.logDir, self.models[k], k, resumeIter, map_location='cuda:0')
        if resumeIter == 'latest':
            resumeIter = -1
        if self.numMpProcess > 0:
            dist.barrier()
        return resumeIter

    def _netFinetuningInitialization(self):
        self.logger.info('[Trainer] MetaModelLoading - _netFinetuningInitialization')
        pass

    def _optimizerSetups(self):
        config = self.config
        self.logger.info('[Trainer] MetaModelLoading - _optimizerSetups')

        # unless the two optimizers are different, you should write in this form
        modelKeys = list([k for k in self.models.keys()
                          if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']])
        params = list(self.models[modelKeys[0]].parameters())
        for j in range(1, len(modelKeys)):
            params += list(self.models[modelKeys[j]].parameters())
        self.optimizerModels = dict()
        self.optimizerModels['all'] = optim.Adam(
            params,
            lr=config.lr,
            eps=config.adam_epsilon,
        )

    def _netHookSetups(self, **kwargs):
        self.logger.info('[Trainer] MetaModelLoading - _netHookSetups')
        hook_type = kwargs['hook_type']
        resumeIter = kwargs['resumeIter']
        config = self.config
        if hook_type is not None:
            assert self.numMpProcess <= 0
            assert self.rank == self.trackedRank
            self.hookModels = {}
            for k in self.models.keys():
                if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                    self.hookModels[k] = PyTorchForwardHook(
                        hook_type, self.models[k],
                        '%s%s%s%sI%d' % (config.P, config.D, config.S, config.R, resumeIter))

    def metaModelLoading(self, **kwargs):
        # kwargs
        iter_label = kwargs['iter_label']  # required for any case
        hook_type = kwargs['hook_type']  # required for any case

        self._netConstruct(f0=float(self.meta['f0']))

        # net train mode (always set to train - if you are doing val, then you should eval it in
        # your function)
        for k in self.models.keys():
            if k != 'meta':
                self.models[k].train()

        self._netRandomInitialization(iter_label=iter_label)
        resumeIter = self._netResumingInitialization(iter_label=iter_label)
        self._netFinetuningInitialization()

        # DDP
        if self.numMpProcess > 0:
            dist.barrier()
            for k in self.models.keys():
                if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                    self.models[k] = torch.nn.parallel.DistributedDataParallel(
                        self.models[k], device_ids=[self.cudaDeviceForAll],
                    )
            dist.barrier()

        self._optimizerSetups()
        self._netHookSetups(hook_type=hook_type, resumeIter=resumeIter)

        return resumeIter

    def metaLossPrepare(self):
        pass

    def setupMetaVariables(self):
        config = self.config
        datasetConf = config.datasetConfDict['corenetChoySingleRendering']

        meta = {}
        meta = self._prepare_corenet_camcube(meta, cudaDevice=self.cudaDeviceForAll)
        if 'nearsurfacemassRForNeighbouringRange' in datasetConf.keys():
            meta = self._prepare_for_voxelizationMasking(
                meta, cudaDevice=self.cudaDeviceForAll, skinNeighbouringName='nearsurfacemass',
                rForNeighbouring=datasetConf['nearsurfacemassRForNeighbouringRange'],
            )
        if 'nearsurfaceairRForNeighbouringRange' in datasetConf.keys():
            meta = self._prepare_for_voxelizationMasking(
                meta, cudaDevice=self.cudaDeviceForAll, skinNeighbouringName='nearsurfaceair',
                rForNeighbouring=datasetConf['nearsurfaceairRForNeighbouringRange'],
            )
        if 'nonsurfaceRForNeighbouringRange' in datasetConf.keys():
            meta = self._prepare_for_voxelizationMasking(
                meta, cudaDevice=self.cudaDeviceForAll, skinNeighbouringName='nonsurface',
                rForNeighbouring=datasetConf['nonsurfaceRForNeighbouringRange'],
            )
        self.meta = meta

    @staticmethod
    def _prepare_for_voxelizationMasking(meta, **kwargs):
        cudaDevice = kwargs['cudaDevice']
        skinNeighbouringName = kwargs['skinNeighbouringName']
        rForNeighbouring = kwargs['rForNeighbouring']

        r = rForNeighbouring
        d = 2 * r + 1
        kernelForNeighbouring = np.ones((1, 1, d, d, d), dtype=np.float32)
        meta['%s_kernelForNeighbouring_thgpu' % skinNeighbouringName] = torch.from_numpy(
            kernelForNeighbouring
        ).to(cudaDevice)
        meta['%s_rForNeighbouring' % skinNeighbouringName] = rForNeighbouring
        return meta

    @staticmethod
    def _prepare_corenet_camcube(meta, **kwargs):
        cudaDevice = kwargs['cudaDevice']
        meta['corenetCamcubeVoxXyzCamZYX_thgpuDict'] = {}
        meta['corenetCamcubeVoxXyzCamPerspZYX_thgpuDict'] = {}

        # meta['corenetSgXCamcubeVoxXyzCamZYX_thgpuDict'] = {}
        # meta['corenetSgXCamcubeVoxXyzCamPerspZYX_thgpuDict'] = {}
        # meta['corenetSgYCamcubeVoxXyzCamZYX_thgpuDict'] = {}
        # meta['corenetSgYCamcubeVoxXyzCamPerspZYX_thgpuDict'] = {}
        # meta['corenetSgZCamcubeVoxXyzCamZYX_thgpuDict'] = {}
        # meta['corenetSgZCamcubeVoxXyzCamPerspZYX_thgpuDict'] = {}

        A1 = udl('pkl_A1b_', 'corenetChoySingleRendering')
        f0 = A1['f0']
        winWidth = A1['winWidth']
        winHeight = A1['winHeight']
        focalLengthWidth = A1['focalLengthWidth']
        focalLengthHeight = A1['focalLengthHeight']
        del A1
        assert len(np.unique(winWidth)) == 1
        assert len(np.unique(winHeight)) == 1
        assert len(np.unique(focalLengthWidth)) == 1
        assert len(np.unique(focalLengthHeight)) == 1
        winWidth0 = float(winWidth[0])
        winHeight0 = float(winHeight[0])
        focalLengthWidth0 = float(focalLengthWidth[0])
        focalLengthHeight0 = float(focalLengthHeight[0])
        delta = 1.e-4
        for s in [128, 64, 32, 16, 8]:
            bound = 0.5 - 0.5 / s
            xi = np.linspace(-bound, bound, s).astype(np.float32)
            yi = np.linspace(-bound, bound, s).astype(np.float32)
            zi = np.linspace(
                f0 / 2 + 0.5 / s, f0 / 2 + 1. - 0.5 / s, s).astype(np.float32)
            x, y, z = np.meshgrid(xi, yi, zi)
            corenetCamcubeVoxXyzCamYXZ = np.stack([
                x, y, z
            ], 3)
            corenetCamcubeVoxXyzCamZYX = \
                corenetCamcubeVoxXyzCamYXZ.transpose((2, 0, 1, 3))
            corenetCamcubeVoxXyzCamPerspZYX = camSys2CamPerspSys0(
                corenetCamcubeVoxXyzCamZYX.reshape((s ** 3, 3)),
                focalLengthWidth0, focalLengthHeight0, winWidth0, winHeight0,
            ).reshape((s, s, s, 3))
            meta['corenetCamcubeVoxXyzCamZYX_thgpuDict'][s] = \
                torch.from_numpy(corenetCamcubeVoxXyzCamZYX).to(cudaDevice)
            meta['corenetCamcubeVoxXyzCamPerspZYX_thgpuDict'][s] = \
                torch.from_numpy(corenetCamcubeVoxXyzCamPerspZYX).to(cudaDevice)

            # for (axis, xyzTag) in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            #     meta['corenetSg%sCamcubeVoxXyzCamZYX_thgpuDict' % xyzTag][s] = \
            #         meta['corenetCamcubeVoxXyzCamZYX_thgpuDict'][s].detach().clone()
            #     meta['corenetSg%sCamcubeVoxXyzCamZYX_thgpuDict' % xyzTag][s][:, :, :, axis] += delta
            #     meta['corenetSg%sCamcubeVoxXyzCamPerspZYX_thgpuDict' % xyzTag][s] = \
            #         camSys2CamPerspSys0(
            #             meta['corenetSg%sCamcubeVoxXyzCamZYX_thgpuDict' % xyzTag][s].reshape(s ** 3, 3),
            #             focalLengthWidth0, focalLengthHeight0, winWidth0, winHeight0,
            #         ).reshape(s, s, s, 3)

        meta['f0'] = f0
        meta['winWidth0'] = winWidth0
        meta['winHeight0'] = winHeight0
        meta['focalLengthWidth0'] = focalLengthWidth0
        meta['focalLengthHeight0'] = focalLengthHeight0
        meta['corenetCamcubeXMin'] = float(-0.5)
        meta['corenetCamcubeXMax'] = float(+0.5)
        meta['corenetCamcubeYMin'] = float(-0.5)
        meta['corenetCamcubeYMax'] = float(+0.5)
        meta['corenetCamcubeZMin'] = float(f0 / 2.)
        meta['corenetCamcubeZMax'] = float(f0 / 2. + 1.)

        # naming:
        # the first Xyz means grid[:, :, :, 0/1/2] means x/y/z coordinate
        # the last YXZ means gird[Y, X, Z, :]
        # if the last YXZ is missing, it is default to be YXZ
        # the first Xyz never missing
        return meta

    def setupMonitor(self):
        self.monitorTrainLogDir = self.logDir + 'monitorTrain/'
        self.monitorValLogDir = self.logDir + 'monitorVal/'
        self.htmlStepperTrain = HTMLStepper(self.monitorTrainLogDir, 100, 'monitorTrain')
        self.htmlStepperVal = HTMLStepper(self.monitorValLogDir, 100, 'monitorVal')
        self.testDataEntry = getTestDataEntryDict(
            wishedTestDataNickName=['corenetSingleOfficialTestSplitFirstOnePerCent'])\
            ['corenetSingleOfficialTestSplitFirstOnePerCent']

    def initializeAll(self, **kwargs):
        # kwargs
        iter_label = kwargs['iter_label']  # required for all cases
        hook_type = kwargs['hook_type']  # required for all cases

        self.setupMetaVariables()
        self.metaDataLoading()
        self.resumeIter = self.metaModelLoading(iter_label=iter_label,
                                                hook_type=hook_type)
        # self.resumeBatch_np = self.metaResumeBatchLoading()
        self.metaLossPrepare()
        self.setupMonitor()

        self.logger.info('Initialization finished! Rank = %d' % self.rank)

    def saveModelSnapshot(self, **kwargs):
        iterCount = kwargs['iterCount']
        if self.rank == self.trackedRank:
            if self.numMpProcess > 0:  # self.net.module
                for k in self.models.keys():
                    if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                        save_network(self.logDir, self.models[k].module, k, str(iterCount))
            else:
                for k in self.models.keys():
                    if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                        save_network(self.logDir, self.models[k], k, str(iterCount))

    def saveBatchVis(self, batch_vis, **kwargs):
        if self.rank == self.trackedRank:
            iterCount = kwargs['iterCount']
            pSaveMat(self.logDir + 'dump/train_iter_%d.mat' % iterCount,
                     {k: batch_vis[k] for k in batch_vis.keys()
                      if not k.startswith('feat') and not k.startswith('fullsurface')})

    def stepMonitorTrain(self, batch_vis, **kwargs):
        iterCount = kwargs['iterCount']
        ifMonitorDump = kwargs['ifMonitorDump']
        ifIsTrackedRank = self.rank == self.trackedRank
        htmlStepper = self.htmlStepperTrain

        config = self.config
        P = config['P']
        D = config['D']
        S = config['S']
        R = config['R']

        testDataEntry = self.testDataEntry
        datasetObj = testDataEntry['datasetObj']

        ben = {}
        batch_vis_now = batchExtractParticularDataset(batch_vis, 'corenetChoySingleRendering')
        for visIndex in range(min(4, batch_vis_now['index'].shape[0])):
            print('    Stepping Visualizer Train: %d' % visIndex)
            assert testDataEntry['datasetObj'].dataset == 'corenetChoySingleRendering'
            bsv0 = testDataEntry['datasetObj'].getOneNP(
                int(batch_vis_now['index'][visIndex]))
            bsv0_initial = constructInitialBatchStepVis0(
                bsv02bsv(bsv0), iterCount=iterCount, visIndex=0, dataset=None,
                P=P, D=D, S=S, R=R,
                verboseGeneral=0,
            )
            bsv0_initial = mergeFromAnotherBsv0(
                bsv0_initial, bsv0,
                copiedKeys=list(set(bsv0.keys()) - set(bsv0_initial.keys()))
            )
            # Treat the testing-on-training as the same way as treating the val cases.
            # So that now we removed the augmentations.

            ifRequiresDrawing = (visIndex < 4) and ifIsTrackedRank
            if True:
                bsv0 = benchmarkingShapenetCorenetFunc(
                    bsv0_initial,
                    datasetObj=datasetObj,
                    # misc
                    cudaDevice=self.cudaDeviceForAll,
                    # benchmarking rules
                    camcubeVoxSize=128,
                    # drawing
                    ifRequiresDrawing=ifIsTrackedRank,
                    # predicting
                    ifRequiresPredictingHere=True,
                    models=self.models, verboseBatchForwarding=0,
                    Trainer=type(self), doPred0Func=self.doPred0,
                    meta=self.meta,
                )
                for k in bsv0.keys():
                    if k.startswith('finalBen'):
                        if k not in ben.keys():
                            ben[k] = []
                        ben[k].append(bsv0[k])
                if ifRequiresDrawing:
                    PDSRI = '%s%s%s%sI%d' % (P, D, S, R, iterCount)
                    summary0, txt0, brInds = addToSummary0Txt0BrInds(
                        summary0=OrderedDict([]), txt0=[], brInds=[0, ],
                        approachEntryDict={PDSRI: {'approachShownName': PDSRI}},
                        bsv0_forVis_dict={PDSRI: bsv0},
                    )
                    headerMessage = '%s-%s-%s-%s-I%d Dataset: %s, Index: %d(%d), visIndex: %d' % \
                        (bsv0['P'], bsv0['D'], bsv0['S'], bsv0['R'], bsv0['iterCount'],
                        bsv0['dataset'], bsv0['index'], bsv0['flagSplit'], bsv0['visIndex'])
                    subMessage = 'CoReNet IoU: %.3f' % bsv0['corenetCubeIou']
                    htmlStepper.step2(summary0, txt0, brInds, headerMessage, subMessage)
                if ifIsTrackedRank and ifMonitorDump and ifRequiresDrawing:
                    for meshName in ['corenetPred', 'corenetLabel']:
                        sysLabel = 'cam'
                        # dumpPly(self.monitorTrainLogDir +
                        #         '%s%s%s%sI%d_%s%s_%s_%d(%d).ply' %
                        #         (P, D, S, R, iterCount, meshName, bt(sysLabel),
                        #          bsv0['dataset'], bsv0['index'], bsv0['flagSplit']),
                        #         bsv0['%sVert%s' % (meshName, bt(sysLabel))],
                        #         bsv0['%sFace' % meshName],
                        #         bsv0['%sVertRgb' % meshName])
            # except:
            #     pass
        ben = {k: np.array(ben[k], dtype=np.float32) for k in ben.keys()}
        return ben

    def stepMonitorVal(self, **kwargs):
        iterCount = kwargs['iterCount']
        ifMonitorDump = kwargs['ifMonitorDump']
        ifIsTrackedRank = self.rank == self.trackedRank
        htmlStepper = self.htmlStepperVal

        config = self.config
        P = config['P']
        D = config['D']
        S = config['S']
        R = config['R']

        testDataEntry = self.testDataEntry
        datasetObj = testDataEntry['datasetObj']
        rank = self.rank
        numMpProcess = self.numMpProcess
        trackedRank = self.trackedRank
        realNumMpProcess = int(max([numMpProcess, 1]))

        indChosen = testDataEntry['indChosen']
        indVisChosen = testDataEntry['indVisChosen']
        j_tot = len(indChosen)

        # re-order
        indNonVisChosen = list(set(indChosen) - set(indVisChosen))
        insertStartingIndex = int(math.floor(float(j_tot) / realNumMpProcess * trackedRank))
        indChosen = indNonVisChosen[:insertStartingIndex] + indVisChosen + \
            indNonVisChosen[insertStartingIndex:]

        if numMpProcess <= 0:
            j1 = 0
            j2 = j_tot
        else:
            j1 = int(math.floor(float(j_tot) / realNumMpProcess * rank))
            j2 = int(math.floor(float(j_tot) / realNumMpProcess * (rank + 1)))
        testDataNickName = testDataEntry['testDataNickName']
        benRecord = {}
        # j2 = j1 + 1  # debug
        for j in range(j1, j2):
            index = indChosen[j]
            print('[Trainer Visualizer Val] Progress Iter%d: '
                  'testDataNickName %s, index = %d, j = %d, j1 = %d, j2 = %d, '
                  'rank = %d, numMpProcess = %d, j_tot = %d' %
                  (iterCount, testDataNickName, index, j, j1, j2,
                   rank, numMpProcess, j_tot))
            batch0_np = testDataEntry['datasetObj'].getOneNP(index)
            batch_np = bsv02bsv(batch0_np)
            batch_vis = batch_np
            bsv0_initial = constructInitialBatchStepVis0(
                batch_vis, iterCount=iterCount, visIndex=0, dataset=None,
                P=P, D=D, S=S, R=R,
                verboseGeneral=0,
            )
            bsv0_initial = mergeFromAnotherBsv0(
                bsv0_initial, bsv2bsv0(batch_vis, visIndex=0),
                copiedKeys=list(set(batch_vis.keys()) - set(bsv0_initial))
            )
            ifRequiresDrawing = (index in indVisChosen) and (rank == trackedRank)
            # step
            # try:
            if True:
                if testDataNickName in ['corenetSingleOfficialTestSplitFirstOnePerCent']:
                    bsv0 = benchmarkingShapenetCorenetFunc(
                        bsv0_initial,
                        datasetObj=datasetObj,
                        # misc
                        cudaDevice=self.cudaDeviceForAll,
                        # benchmarking rules
                        camcubeVoxSize=128,
                        # drawing
                        ifRequiresDrawing=ifIsTrackedRank,
                        # predicting
                        ifRequiresPredictingHere=True,
                        models=self.models, verboseBatchForwarding=0,
                        Trainer=type(self), doPred0Func=self.doPred0,
                        meta=self.meta,
                    )
                else:
                    raise NotImplementedError('Unknown testDataNickName: %s' % testDataNickName)
                tmp = [k for k in bsv0.keys() if k.startswith('predBen')]
                for k in tmp:
                    _ = k[len('predBen'):]
                    bsv0['finalBen%s' % bt(_)] = bsv0['predBen%s' % bt(_)]

                bsv0_toStore = {}
                for k in bsv0.keys():
                    if k.startswith('finalBen') or k in [
                        'iterCount', 'visIndex', 'P', 'D', 'S', 'R', 'methodologyName',
                        'index', 'did', 'datasetID', 'dataset', 'flagSplit',
                    ] or k.startswith('finalBen'):
                        bsv0_toStore[k] = bsv0[k]
                benRecord[j] = bsv0_toStore

                if ifRequiresDrawing:
                    PDSRI = '%s%s%s%sI%d' % (P, D, S, R, iterCount)
                    summary0, txt0, brInds = addToSummary0Txt0BrInds(
                        summary0=OrderedDict([]), txt0=[], brInds=[0, ],
                        approachEntryDict={PDSRI: {'approachShownName': PDSRI}},
                        bsv0_forVis_dict={PDSRI: bsv0},
                    )
                    headerMessage = '%s-%s-%s-%s-I%d Dataset: %s, Index: %d(%d), visIndex: %d' % \
                                    (bsv0['P'], bsv0['D'], bsv0['S'], bsv0['R'], bsv0['iterCount'],
                                     bsv0['dataset'], bsv0['index'], bsv0['flagSplit'], bsv0['visIndex'])
                    subMessage = 'CorenetCubeIou: %.3f' % bsv0['corenetCubeIou']
                    htmlStepper.step2(summary0, txt0, brInds, headerMessage, subMessage)
                if ifMonitorDump and ifIsTrackedRank and ifRequiresDrawing:
                    for meshName in ['corenetPred', 'corenetLabel']:
                        sysLabel = 'cam'
                        dumpPly(self.monitorValLogDir +
                                '%s%s%s%sI%d_%s%s_%s_%d(%d).ply' %
                                (P, D, S, R, iterCount, meshName, bt(sysLabel),
                                 bsv0['dataset'], bsv0['index'], bsv0['flagSplit']),
                                bsv0['%sVert%s' % (meshName, bt(sysLabel))],
                                bsv0['%sFace' % meshName],
                                bsv0['%sVertRgb' % meshName])
            # except Exception:
            #     pass

        ar = []
        keys = [k[len('finalBen'):] for k in list(benRecord.values())[0].keys() if k.startswith('finalBen')]
        for br in benRecord.values():
            ar.append(np.array([
                br['finalBen%s' % bt(k)] for k in keys
            ], dtype=np.float32))
        ar = np.stack(ar, 1)
        arSum = ar.sum(1)
        arSum = np.concatenate([arSum, np.array([len(benRecord)], dtype=np.float32)], 0)
        arSum_thgpu = torch.from_numpy(arSum).to(self.cudaDeviceForAll)
        if numMpProcess > 0:
            dist.barrier()
            dist.all_reduce(arSum_thgpu, op=ReduceOp.SUM)
            dist.barrier()
        validCount = int(arSum_thgpu[-1].detach().cpu().numpy())
        assert validCount > 0, 'validCount is 0'
        arMean = arSum_thgpu.detach().cpu().numpy() / float(validCount)
        ben = {'finalBen%s' % bt(keys[j]): np.array(arMean[j], dtype=np.float32)
               for j in range(len(keys))}
        ben['finalBenValidCount'] = np.array(validCount, dtype=np.float32)
        return ben

    def stepBatch(self, **kwargs):
        config = self.config
        tmp = OrderedDict([])
        for did, (dataset, datasetConf) in enumerate(config.datasetConfDict.items()):
            try:
                batchDid_thcpu = next(self.dataLoaderIterDict[dataset])
            except:
                self.dataLoaderIterDict[dataset] = iter(self.dataLoaderDict[dataset])
                batchDid_thcpu = next(self.dataLoaderIterDict[dataset])
            tmp[dataset] = batchDid_thcpu

        batch_thcpu = {}
        for dataset in tmp.keys():
            for k in tmp[dataset].keys():
                batch_thcpu[k + '_' + dataset] = tmp[dataset][k]

        batch_thgpu = castAnything(batch_thcpu, 'thcpu2thgpu', device=self.cudaDeviceForAll)
        return batch_thcpu, batch_thgpu

    @staticmethod
    def _generalVoxelizationWithSkin(faceVertZO0List, Ly, Lx, Lz, cudaDevice):
        # ZO: Zero To One
        # Note this ZO cam sys is the corenet cube in the cam sys
        # It is different from camTheirs in the _voxelization() method
        # for the y dim they are summed to 1 (complementary)
        # so in this method, we should not reverse the y dim in the output.

        faceVertZO0s = np.concatenate(faceVertZO0List, 0)
        nFaces = np.array([faceVertZO0.shape[0] for faceVertZO0 in faceVertZO0List],
                          dtype=np.int32)
        grids_thcpu = voxelize_mesh(
            torch.from_numpy(faceVertZO0s),
            torch.from_numpy(nFaces),
            resolution=(Lx, Ly, Lz),
            view2voxel=torch.from_numpy(
                np.array([
                    [Lx, 0, 0, 0],
                    [0, Ly, 0, 0],
                    [0, 0, Lz, 0],
                    [0, 0, 0, 1],
                ], dtype=np.float32),
            ),
            sub_grid_sampling=False,
            image_resolution_multiplier=8,
            conservative_rasterization=False,
            projection_depth_multiplier=1,
            cuda_device=None,
        )  # (BZYX)
        grids_thgpu = grids_thcpu.to(cudaDevice)
        skins_thgpu = grids_thgpu.detach().clone()
        fill_voxels.fill_inside_voxels_gpu(grids_thgpu, inplace=True)
        return grids_thgpu, skins_thgpu  # (BZYX) bisem: 1 for the occupied

    @classmethod
    def samplingBatchForCorenetTHGPU(cls, batch_thcpu, batch_thgpu, **kwargs):
        meta = kwargs['meta']
        datasetObj = kwargs['datasetObj']
        datasetConf = kwargs['datasetConf']

        batchSizePerProcess = datasetConf['batchSizePerProcess']

        f0 = meta['f0']

        faceVertZO0List = []
        for j in range(batch_thcpu['index_corenetChoySingleRendering'].shape[0]):
            index = int(batch_thcpu['index_corenetChoySingleRendering'][j])
            tmp1 = datasetObj.getRawMeshCamOurs({}, index)
            face0 = tmp1['face'].astype(np.int32)
            tmp = tmp1['vertCamOurs'].astype(np.float32)
            tmp[:, :2] += 0.5
            tmp[:, -1] -= f0 / 2.
            vertZO0 = tmp
            faceVertZO0List.append(vertInfo2faceVertInfoNP(vertZO0[None], face0[None])[0])

        corenetCamcubeLabelBisemZYX, corenetCamcubeLabelSkinZYX = cls._generalVoxelizationWithSkin(
            faceVertZO0List,
            128, 128, 128, cudaDevice=batch_thgpu['index_corenetChoySingleRendering'].device)

        if 'samplingMethodList' not in datasetConf.keys():
            samplingMethodList = ['gridpoint']
        else:
            samplingMethodList = datasetConf['samplingMethodList']
        for samplingMethod in samplingMethodList:
            if samplingMethod == 'gridpoint':
                numSamplingFinalGridpoint = datasetConf['numSamplingFinalGridpoint']
                ra_gridpoint = torch.randperm(
                    128 ** 3, device=batch_thgpu['index_corenetChoySingleRendering'].device)[:numSamplingFinalGridpoint]
                batch_thgpu['finalGridpointPCxyzCam_corenetChoySingleRendering'] = \
                    meta['corenetCamcubeVoxXyzCamZYX_thgpuDict'][128].reshape(
                        1, -1, 3)[:, ra_gridpoint, :].repeat(batchSizePerProcess, 1, 1).detach()
                batch_thgpu['finalGridpointPClabelBisem_corenetChoySingleRendering'] = corenetCamcubeLabelBisemZYX.reshape(
                    batchSizePerProcess, 128 ** 3)[:, ra_gridpoint].detach()
                del numSamplingFinalGridpoint, ra_gridpoint

            elif samplingMethod in ['nearsurfacemass', 'nearsurfaceair', 'nonsurface']:
                numSampling = datasetConf['numSamplingFinal%s' % bt(samplingMethod)]
                kernelForNeighbouring = meta['%s_kernelForNeighbouring_thgpu' % samplingMethod]
                rForNeighbouring = meta['%s_rForNeighbouring' % samplingMethod]
                gridPurtFactor = datasetConf['%sGridPurtFactor' % samplingMethod]
                corenetCamcubeLabelSkinNeighbouringZYX = (F.conv3d(
                    input=corenetCamcubeLabelSkinZYX[:, None, :, :, :],
                    weight=kernelForNeighbouring, bias=None, stride=1,
                    padding=rForNeighbouring,
                )[:, 0, :, :, :] > 0).float()
                # flagging
                if samplingMethod == 'nearsurfacemass':
                    valid = (corenetCamcubeLabelSkinNeighbouringZYX *
                             corenetCamcubeLabelBisemZYX).reshape(batchSizePerProcess, -1)
                elif samplingMethod == 'nearsurfaceair':
                    valid = (corenetCamcubeLabelSkinNeighbouringZYX *
                             (1 - corenetCamcubeLabelBisemZYX)).reshape(batchSizePerProcess, -1)
                elif samplingMethod == 'nonsurface':
                    valid = (1 - corenetCamcubeLabelSkinNeighbouringZYX).reshape(batchSizePerProcess, -1)
                else:
                    raise NotImplementedError('Unknown samplingMethod: %s' % samplingMethod)
                batch_thgpu['%sMaskRate_corenetChoySingleRendering' % samplingMethod] = valid.float().sum(1) / valid.shape[1]
                # bubble up
                tmp = torch.argsort(
                    valid * (1 + torch.rand(  # 1+ is important, as sometimes your rand returns 0.
                        batchSizePerProcess, valid.shape[1],
                        dtype=torch.float32, device=valid.device)),
                    dim=1, descending=True)
                #    introducing the torch.random mainly to enable random argsort for equal values (valid == 1).
                cutoff = int(valid.sum(1).min())
                tmp = tmp[:, :cutoff]
                ra_for_all_samples = torch.arange(end=cutoff, dtype=torch.int64, device=None)
                while ra_for_all_samples.shape[0] < numSampling:
                    ra_for_all_samples = torch.cat([ra_for_all_samples, ra_for_all_samples], 0)
                ra_for_all_samples = ra_for_all_samples[:numSampling]
                #    up to here, we pick numSampling queries from the valid pool for each sample.
                tmp = tmp[:, ra_for_all_samples]
                tmp3 = tmp[:, :, None].repeat(1, 1, 3)
                # gather
                pcXyzCam = torch.gather(
                    meta['corenetCamcubeVoxXyzCamZYX_thgpuDict'][128].reshape(
                        1, -1, 3).repeat(batchSizePerProcess, 1, 1),
                    dim=1, index=tmp3)
                pcXyzCam += (2. * torch.rand(
                    pcXyzCam.shape[0], pcXyzCam.shape[1], pcXyzCam.shape[2],
                    dtype=torch.float32, device=pcXyzCam.device) - 1.) * \
                            (gridPurtFactor * 0.5 / 128)
                #    verification purpose
                pcLabelBisem = torch.gather(
                    corenetCamcubeLabelBisemZYX.view(batchSizePerProcess, -1), dim=1, index=tmp)
                if samplingMethod == 'nearsurfacemass':
                    assert (pcLabelBisem == 1).all()
                elif samplingMethod == 'nearsurfaceair':
                    assert (pcLabelBisem == 0).all()
                elif samplingMethod == 'nonsurface':
                    pass  # check nothing
                else:
                    raise NotImplementedError('Unknown samplingMethod: %s' % samplingMethod)
                valid_recheck = torch.gather(valid, dim=1, index=tmp)
                if not (valid_recheck == 1).all():
                    import ipdb
                    ipdb.set_trace()
                assert (valid_recheck == 1).all()

                # store
                batch_thgpu['final%sPCxyzCam_corenetChoySingleRendering' % bt(samplingMethod)] = pcXyzCam
                batch_thgpu['final%sPClabelBisem_corenetChoySingleRendering' % bt(samplingMethod)] = pcLabelBisem
            else:
                raise NotImplementedError('Unknown samplingMethod: %s' % samplingMethod)

        batch_thgpu['corenetCamcubeLabelBisemZYX_corenetChoySingleRendering'] = corenetCamcubeLabelBisemZYX

        return batch_thgpu

    def batchPreprocessingTHGPU(self, batch_thcpu, batch_thgpu, **kwargs):
        config = self.config
        iterCount = kwargs['iterCount']

        batch_thgpu = self.samplingBatchForCorenetTHGPU(
            batch_thcpu,
            batch_thgpu,
            datasetObj=self.datasetObjDict['corenetChoySingleRendering'],
            datasetConf=config.datasetConfDict['corenetChoySingleRendering'],
            cudaDevice=self.cudaDeviceForAll,
            verbose=self.samplingVerbose,
            meta=self.meta,
            projRoot=self.projRoot,
            iterCount=iterCount)
        return batch_thgpu

    @staticmethod
    def assertModelsTrainingMode(models):
        for k, v in models.items():
            if (k not in models['meta']['nonLearningModelNameList']) and (k != 'meta'):
                assert v.training

    @staticmethod
    def assertModelEvalMode(models):
        for k, v in models.items():
            if (k not in models['meta']['nonLearningModelNameList']) and (k != 'meta'):
                assert not v.training

    @staticmethod
    def setModelsTrainingMode(models):
        for k, v in models.items():
            if (k not in models['meta']['nonLearningModelNameList']) and (k != 'meta'):
                v.train()

    @staticmethod
    def setModelsEvalMode(models):
        for k, v in models.items():
            if (k not in models['meta']['nonLearningModelNameList']) and (k != 'meta'):
                v.eval()

    @classmethod
    def doQueryPred0(cls, img0, queryPointCam0, queryPointCamPersp0,
                     **kwargs):

        meta = kwargs['meta']
        models = kwargs['models']
        cudaDevice = kwargs['cudaDevice']
        batchSize = kwargs.get('batchSize', 100 ** 2)
        verboseBatchForwarding = kwargs.get('verboseBatchForwarding', 0)
        samplingFromNetwork1ToNetwork2Func = cls._samplingFromNetwork1ToNetwork2
        samplingFromNetwork2ToNetwork3Func = cls._samplingFromNetwork2ToNetwork3

        assert len(queryPointCam0.shape) == 2 and queryPointCam0.shape[1] == 3
        assert len(queryPointCamPersp0.shape) == 2 and queryPointCamPersp0.shape[1] == 3
        assert len(img0) == 3 and img0.shape[0] == 3

        corenetCamcubeVoxXyzCamPerspZYX_thgpuDict = \
            meta['corenetCamcubeVoxXyzCamPerspZYX_thgpuDict']
        f0 = meta['f0']

        encoder = models['encoder']
        networkTwo = models['networkTwo']
        encoder.eval()
        networkTwo.eval()

        nQuery = queryPointCam0.shape[0]
        assert nQuery == queryPointCamPersp0.shape[0]
        batchTot = int(math.ceil((float(nQuery) / batchSize)))

        # encoder (Network One)
        image_initial_thgpu = torch.from_numpy(img0[None, :, :, :]).to(cudaDevice)
        image_thgpu = resnet50.preprocess_image_caffe(
            (image_initial_thgpu * 255.).byte())
        ss = encoder(image_thgpu)

        # sampling from Network One
        tt = samplingFromNetwork1ToNetwork2Func(
            ss, corenetCamcubeVoxXyzCamPerspZYX_thgpuDict,
            batchSizeTotPerProcess=1)

        # Network Two
        yy = networkTwo(tt)
        yy['y6'] = torch.sigmoid(yy['y6'])

        predBisem0 = np.zeros((nQuery,), dtype=np.float32)
        for batchID in range(batchTot):
            if verboseBatchForwarding > 0 and (batchID < 5 or batchID % verboseBatchForwarding == 0):
                print('    Processing doQueryPred for batches %d / %d' % (batchID, batchTot))
            head = batchID * batchSize
            tail = min((batchID + 1) * batchSize, nQuery)

            pCam = torch.from_numpy(
                queryPointCam0[head:tail, :]
            ).to(cudaDevice)[None, :, :]

            zz = samplingFromNetwork2ToNetwork3Func(
                yy, pCam, f0=f0,
            )

            out = zz['z6'].reshape(-1)
            predBisem0[head:tail] = out.detach().cpu().numpy()

        predOccfloat0 = 1. - predBisem0
        return {'occfloat': predOccfloat0, 'bisem': predBisem0}

    @classmethod
    def doPred0(cls, bsv0, **kwargs):
        meta = kwargs['meta']
        models = kwargs['models']
        cudaDevice = kwargs['cudaDevice']
        Ly, Lx, Lz, _ = bsv0['corenetCamcubeVoxXyzCam'].shape
        tmp0 = cls.doQueryPred0(
            bsv0['img'],
            bsv0['corenetCamcubeVoxXyzCam'].reshape((Ly * Lx * Lz, 3)),
            bsv0['corenetCamcubeVoxXyzCamPersp'].reshape((Ly * Lx * Lz, 3)),
            meta=meta, models=models, cudaDevice=cudaDevice,
        )
        fullQueryMaskfloat0 = bsv0['corenetCamcubeMaskfloat'].reshape((Ly * Lx * Lz))
        fullQueryOccfloat0 = tmp0['occfloat']
        fullQueryOccfloat0[fullQueryMaskfloat0 <= 0] = 1.  # empty
        fullQueryBisem0 = tmp0['bisem']
        fullQueryBisem0[fullQueryMaskfloat0 <= 0] = 0.  # empty

        bsv0['corenetCamcubePredOccfloatYXZ'] = fullQueryOccfloat0.reshape((Ly, Lx, Lz))
        bsv0['corenetCamcubePredBisemYXZ'] = fullQueryBisem0.reshape((Ly, Lx, Lz))

        return bsv0

    @classmethod
    def postprocess1(cls, tmp0, tmp1, **kwargs):
        cudaDevice = kwargs['cudaDevice']
        d = grid_sample(
            torch.from_numpy(tmp1['depthPred'][None, None, :, :]).float().to(cudaDevice),
            torch.from_numpy(tmp0['extractedXyzCamPersp'][None, :, None, :2]).to(cudaDevice),
            mode='nearest', padding_mode='zeros', align_corners=False,
        ).detach().cpu().numpy()[0, 0, :, 0]
        z = tmp0['extractedXyzCam'][:, 2]
        Ly, Lx, Lz = tmp0['Ly'], tmp0['Lx'], tmp0['Lz']
        o1 = np.ones_like(tmp1['occfloat'].copy())
        o1[z < d] = 1.
        thre = 0.2

        assert thre >= 0.5 * tmp0['sCell'][2]
        t = (z >= d - 0.5 * tmp0['sCell'][2]) & (z < d + thre)
        zt = z[t]
        dt = d[t]
        nt = np.floor((dt - tmp0['goxyzCam'][2]) / tmp0['sCell'][2] + 0.5)
        z0t = tmp0['goxyzCam'][2] + tmp0['sCell'][2] * (nt - 0.5)
        z1t = tmp0['goxyzCam'][2] + tmp0['sCell'][2] * (nt + 0.5)
        flagHot = (z0t <= zt) & (zt <= z1t)
        kk = o1[t]
        kk[flagHot] = 0.5
        kk[flagHot == 0] = 0.
        o1[t] = kk

        o1[z > d + thre] = 2.
        oo1 = np.ones((Ly * Lx * Lz), dtype=np.float32)
        oo1[tmp0['flagQuery']] = o1
        oo1 = oo1.reshape((Ly, Lx, Lz))
        # cc3d
        t = cc3d.connected_components(1 - oo1.astype(np.int32), connectivity=6)
        for i in np.unique(t):
            if (t == i).sum() < 100:
                oo1[t == i] = 1.

        occfloat0 = np.ones((tmp0['Lx'] * tmp0['Ly'] * tmp0['Lz'],), dtype=np.float32)
        occfloat0[tmp0['flagQuery']] = tmp1['occfloat']
        occfloat0 = occfloat0.reshape((tmp0['Ly'], tmp0['Lx'], tmp0['Lz']))
        occfloat0[oo1 <= 1] = oo1[oo1 <= 1]

        if occfloat0.min() > 0.5:
            occfloat0[0] = 0.
        if occfloat0.max() < 0.5:
            occfloat0[-1] = 1.
        predVertCam0, predFace0 = voxSdfSign2mesh_skmc(
            voxSdfSign=occfloat0, goxyz=tmp0['goxyzCam'], sCell=tmp0['sCell'],
        )
        return predVertCam0, predFace0

    @classmethod
    def _samplingFromNetwork1ToNetwork2(cls, ss, corenetCamcubeVoxXyzCamPerspZYX_thgpuDict, **kwargs):
        B = ss['s2'].shape[0]
        g = lambda s, l3: grid_sample(
            s,
            corenetCamcubeVoxXyzCamPerspZYX_thgpuDict[l3][:, :, :, :2].reshape(
                1, l3 ** 3, 1, 2).repeat(B, 1, 1, 1, ),
            mode='bilinear', padding_mode='border', align_corners=False,
        ).reshape((B, s.shape[1], l3, l3, l3))
        tt = {}
        tt['ta'] = ss['sa']
        for j in [2, 3, 4, 5]:
            if 's%d' % j in ss.keys():
                tt['t%d' % j] = g(ss['s%d' % j], int(2 ** (8 - j)))
        return tt

    @classmethod
    def _samplingFromNetwork2ToNetwork3(cls, yy, pointPCxyzCam, f0, **kwargs):
        zzUsedKeys = ['z6']
        h = lambda y: grid_sample(
            y,
            torch.stack([
                pointPCxyzCam[:, :, 0] * 2.,  # x from [-0.5, 0.5] to [-1, 1]
                pointPCxyzCam[:, :, 1] * 2.,  # y from [-0.5, 0.5] to [-1, 1]
                (pointPCxyzCam[:, :, 2] - f0 / 2. - 0.5) * 2.,  # z from [f0/2, 1 + f0/2] to [-1, 1]
            ], 2)[:, :, None, None, :],
            mode='bilinear', padding_mode='border', align_corners=False,
        ).reshape((pointPCxyzCam.shape[0], y.shape[1], pointPCxyzCam.shape[1]))
        zz = {}
        if 'y0' in yy.keys():
            zz['z0'] = yy['y0'][:, :, None].repeat(1, 1, pointPCxyzCam.shape[1])
        for j in [1, 2, 3, 4, 5, 6]:
            if 'z%d' % j in zzUsedKeys:
                zz['z%d' % j] = h(yy['y%d' % j])
        return zz

    @classmethod
    def forwardNetForCorenet(cls, batch_thgpu, **kwargs):
        models = kwargs['models']
        meta = kwargs['meta']

        corenetCamcubeVoxXyzCamPerspZYX_thgpuDict = \
            meta['corenetCamcubeVoxXyzCamPerspZYX_thgpuDict']

        encoder = models['encoder']
        networkTwo = models['networkTwo']

        # network 1
        image_initial_thgpu = batch_thgpu['img_corenetChoySingleRendering']
        image_thgpu = resnet50.preprocess_image_caffe(
            (image_initial_thgpu * 255.).byte())
        ss = encoder(image_thgpu)

        # sampling from network 1
        tt = cls._samplingFromNetwork1ToNetwork2(
            ss, corenetCamcubeVoxXyzCamPerspZYX_thgpuDict,
            batchSizeTotPerProcess=int(image_thgpu.shape[0]),
        )

        # network 2
        yy = networkTwo(tt)
        yy['y6'] = torch.sigmoid(yy['y6'])
        batch_thgpu['corenetCamcubePredBisemZYX_corenetChoySingleRendering'] = yy['y6'][:, 0, :, :, :]

        for samplingMethod in ['gridpoint']:
            pCam = batch_thgpu['final%sPCxyzCam_corenetChoySingleRendering' % bt(samplingMethod)]

            zz = cls._samplingFromNetwork2ToNetwork3(
                yy, pCam,
                f0=meta['f0'],
            )

            batch_thgpu['final%sPCpredBisem_corenetChoySingleRendering' % bt(samplingMethod)] = \
                zz['z6'].view(pCam.shape[0], pCam.shape[1])
        return batch_thgpu

    @classmethod
    def forwardLoss(cls, batch_thgpu, **kwargs):
        wl = kwargs['wl']

        # lossGridpointIou
        if 'lossGridpointIou' in wl.keys() and wl['lossGridpointIou'] > 0:
            pred = batch_thgpu['finalGridpointPCpredBisem_corenetChoySingleRendering']
            label = batch_thgpu['finalGridpointPClabelBisem_corenetChoySingleRendering']
            B = label.shape[0]
            iou = torch.div(
                torch.min(pred, label).reshape(B, -1).sum(1),
                torch.max(pred, label).reshape(B, -1).sum(1) + 1.e-4,
            )
            batch_thgpu['lossGridpointIou'] = (1. - iou).mean()
            del pred, label, B, iou

        # statIou
        pred = batch_thgpu['corenetCamcubePredBisemZYX_corenetChoySingleRendering'] > 0.5
        label = batch_thgpu['corenetCamcubeLabelBisemZYX_corenetChoySingleRendering']
        B = label.shape[0]
        iou = torch.div(
            torch.min(pred, label).reshape(B, -1).sum(1),
            torch.max(pred, label).reshape(B, -1).sum(1) + 1.e-4,
        )
        batch_thgpu['statIou'] = iou.mean()
        del pred, label, B, iou

        # weighted
        for k in batch_thgpu.keys():
            if k.startswith('loss'):
                batch_thgpu[k] *= wl[k]

        # loss total
        batch_thgpu['loss'] = sum([batch_thgpu[k] for k in batch_thgpu.keys()
                                   if k.startswith('loss')])

        return batch_thgpu

    def forwardBackwardUpdate(self, batch_thgpu, **kwargs):
        ifAllowBackward = kwargs['ifAllowBackward']
        iterCount = kwargs['iterCount']
        ifRequiresGrad = kwargs['ifRequiresGrad']

        config = self.config

        batch_thgpu = self.forwardNetForCorenet(
            batch_thgpu,
            models=self.models,
            meta=self.meta,
            datasetConf=self.config.datasetConfDict['corenetChoySingleRendering'],
            ifRequiresGrad=ifRequiresGrad,
        )
        batch_thgpu = self.forwardLoss(
            batch_thgpu,
            wl=config.wl,
        )
        if ifAllowBackward:
            self.backwardLoss(
                batch_thgpu, iterCount=iterCount, optimizerModels=self.optimizerModels,
            )
        return batch_thgpu

    @classmethod
    def backwardLoss(cls, batch_thgpu, **kwargs):
        iterCount = kwargs['iterCount']
        optimizerModels = kwargs['optimizerModels']

        if iterCount > 0:
            optimizerModels['all'].zero_grad()
            batch_thgpu['loss'].backward()
            optimizerModels['all'].step()
        return batch_thgpu

    def trainNoException(self, **kwargs):
        config = self.config
        R = config.R
        if self.resumeIter >= 0:
            iterCount = self.resumeIter
        else:
            iterCount = 0
        while True:
            timeIterStart = time.time()

            # Rtmp check
            if iterCount == 100 and config.R.startswith('Rtmp'):
                self.logger.info(
                    'This is %s session, which is temporary!!! No Further Forwarding' % config.R)
                return

            # whether or not to dump
            power10 = int(math.log10(float(iterCount + 1)))
            power10 = min(power10, 5)
            divider = 10 ** power10

            # these ifs does not take the rank / numMpProcess into account
            monitorMode = self.monitorMode
            ifStore = iterCount > self.resumeIter and (
                    iterCount % max([divider, self.minSnapshotFreq]) == monitorMode)
            # or iterCount == self.resumeIter + 1)
            ifBackupTheLatestModels = (iterCount % 10000 == 0) and (iterCount > 100000)
            ifMonitorDump = iterCount % self.monitorDumpFreq == monitorMode
            ifMonitorTrain = (iterCount > self.resumeIter) and (
                    (iterCount % self.monitorTrainFreq == monitorMode) or
                    (iterCount < self.monitorTrainFreq and iterCount % (
                            self.monitorTrainFreq / 2) == monitorMode))
            ifMonitorTrain = ifMonitorTrain or ifMonitorDump
            ifMonitorVal = (iterCount > self.resumeIter) and (
                    (iterCount % self.monitorValFreq == monitorMode) or
                    (iterCount < self.monitorValFreq and iterCount % (self.monitorValFreq / 2) == monitorMode))
            ifMonitorVal = (ifMonitorVal or ifMonitorDump) and \
                           ((iterCount >= self.monitorValFreq) or (R.startswith('Rtmpval')))

            ifPrint = ifStore or (iterCount - self.resumeIter) % self.printFreq == 0 or iterCount < 20
            if ifMonitorTrain or ifMonitorVal:
                ifPrint = True
            # Note ifStore do not do this - all the threads needs sync. ifPrint does not relate to sync.
            ifSaveToFinishedIter = iterCount % 50 == 0

            # ifTracked
            ifTracked = self.rank == self.trackedRank or self.numMpProcess == 0

            # printing
            if ifPrint and ifTracked:
                self.logger.info(
                    '---------- Iter %d Training: Host %s GPU %s CPU %d '
                    'Rank %d NumMpProcess %d %s %s %s %s ---------' %
                    (iterCount, gethostname(),
                     os.environ['CUDA_VISIBLE_DEVICES'], os.getpid(),
                     self.rank, self.numMpProcess, config.P,
                     config.D, config.S,
                     config.R))
                self.logger.info('    [TimeStamp] timeStamp Iter%d: ' % iterCount +
                                 time.strftime('%m/%d/%y %H:%M:%S', time.localtime()))

            # -------------------------------- batch data loading -------------------------------- #
            t = time.time()

            batch_thcpu, batch_thgpu = self.stepBatch()

            if ifPrint and ifTracked:
                self.logger.info('    [Timer] dataLoading Iter%d: %.3f seconds' % (iterCount, time.time() - t))

            # ----------------------------------- Preprocessing ----------------------------------- #
            t = time.time()
            batch_thgpu = self.batchPreprocessingTHGPU(
                batch_thcpu, batch_thgpu,
                datasetObjDict=self.datasetObjDict,
                iterCount=iterCount)
            if ifPrint and ifTracked:
                self.logger.info(
                    '    [Timer] batchPreprocessingTHGPU Iter%d: %.3f seconds' % (iterCount, time.time() - t))

            # ------------------------------------ main course ------------------------------------ #
            t = time.time()

            batch_thgpu = self.forwardBackwardUpdate(
                batch_thgpu,
                ifTrain=True,
                iterCount=iterCount,
                ifAllowBackward=True,
                ifRequiresGrad=True)
            if ifPrint and ifTracked:
                self.logger.info(
                    '    [Timer] forwardBackwardUpdate Iter%d: %.3f seconds' % (iterCount, time.time() - t))

            # ---------------------------------------- Meta --------------------------------------- #
            # Storing
            if ifSaveToFinishedIter and ifTracked:
                self.finishedIterCount = iterCount
                self.finishedBatchVis = castAnything(batch_thgpu, 'thgpu2np')

            if ifStore and ifTracked:
                # save model snapshot
                self.saveModelSnapshot(iterCount=iterCount)

                # visualTrain direct saving (not including visualization)
                batch_vis = castAnything(batch_thgpu, 'thgpu2np')
                self.saveBatchVis(batch_vis, iterCount=iterCount)
                # del batch_thgpu

                # visualVal forwarding and saving (not including visualization)
                # self.saveValBatchVis(iterCount=iterCount)
            else:
                # del batch_thgpu
                pass

            if ifBackupTheLatestModels and ifTracked:
                self.saveModelSnapshot(iterCount='latest')

            # MonitorTrain
            if ifMonitorTrain:
                if self.numMpProcess > 0:
                    dist.barrier()

                with torch.no_grad():
                    self.setModelsEvalMode(self.models)
                    if self.numMpProcess > 0:
                        dist.barrier()
                    batch_vis = castAnything(batch_thgpu, 'thgpu2np')
                    ben = self.stepMonitorTrain(batch_vis, iterCount=iterCount,
                                                ifMonitorDump=ifMonitorDump,
                                                )
                    for k in ben.keys():
                        batch_thgpu['statTrain' + bt(k)] = \
                            torch.nanmean(torch.from_numpy(ben[k]).to(self.cudaDeviceForAll))
                    if self.numMpProcess > 0:
                        dist.barrier()
                    self.setModelsTrainingMode(self.models)

                if self.numMpProcess > 0:
                    dist.barrier()

            # MonitorVal
            if ifMonitorVal:
                if self.numMpProcess > 0:
                    dist.barrier()

                with torch.no_grad():
                    self.setModelsEvalMode(self.models)
                    if self.numMpProcess > 0:
                        dist.barrier()
                    ben = self.stepMonitorVal(iterCount=iterCount,
                                              ifMonitorDump=ifMonitorDump,
                                              )
                    for k in ben.keys():
                        batch_thgpu['statVal' + bt(k)] = \
                            torch.nanmean(torch.from_numpy(ben[k]).to(self.cudaDeviceForAll))
                    if self.numMpProcess > 0:
                        dist.barrier()
                    self.setModelsTrainingMode(self.models)

                if self.numMpProcess > 0:
                    dist.barrier()

            # Print and Log the Loss and the Benchmarking
            if ifPrint and ifTracked:
                for k in batch_thgpu.keys():
                    if k.startswith('loss'):
                        self.logger.info('    [Loss] %s Iter%d: %.5f' %
                                         (k, iterCount, float(batch_thgpu[k])))
                    if k.startswith('stat'):
                        self.logger.info('    [Stat] %s Iter%d: %.5f' %
                                         (k, iterCount, float(batch_thgpu[k])))

            # Print and Log the Time
            if ifPrint and ifTracked:
                self.logger.info('[Timer] Total is %.3f seconds.' % (time.time() - timeIterStart))

            iterCount += 1
            del batch_thgpu
            pass

    def train(self, **kwargs):
        # self.trainNoException()
        try:
            self.trainNoException(**kwargs)
        except KeyboardInterrupt as e:
            self.logger.info('KeyboardInterrupt Detected: Rank %d' % self.rank)
            if self.rank == self.trackedRank and self.finishedIterCount > 10000:
                self.saveModelSnapshot(iterCount=self.finishedIterCount)
                self.saveBatchVis(self.finishedBatchVis, iterCount=self.finishedIterCount)
                # self.saveValBatchVis(iterCount=self.finishedIterCount)
            dist.barrier()
            dist.destroy_process_group()


