# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Basic
import torch
import torch.nn as nn
import torch.distributions
import torch.nn.functional as F
from torch.nn.functional import grid_sample
import torchvision
import torch.optim as optim
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

# toolbox
from UDLv3 import udl
from codes_py.toolbox_framework.framework_util_v2 import DatasetObjKeeper, DataLoaderKeeper, \
    checkPDSRLogDirNew, castAnything, probe_load_network, load_network, save_network
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
from codes_py.np_ext.data_processing_utils_v1 import determinedArbitraryPermutedVector2

# PDSR
from .dataset import CorenetChoySingleRenderingDataset
from . import resnet50
from .decoder import Decoder
from . import file_system as fs

# visualizer from configs_registration
from configs_registration import getConfigGlobal

# external_codes
from codes_py.corenet.geometry.voxelization import voxelize_mesh
from codes_py.corenet.cc import fill_voxels

# inlines
bt = lambda s: s[0].upper() + s[1:]


class Trainer(object):
    def __init__(self, config, **kwargs):
        self.ifDumpLogging = kwargs['ifDumpLogging']

        # logDir
        self.projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
        self.logDir = checkPDSRLogDirNew(config,
                                         projRoot=self.projRoot,
                                         ifInitial=config.R.startswith('Rtmp'))

        # logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)
        fh = logging.FileHandler(self.logDir + 'trainingLog.txt')
        fh.setLevel(logging.INFO)
        if self.ifDumpLogging:
            self.logger.addHandler(fh)
        self.logger.propagate = False
        self.logger.info('====== Starting training: GPU %s CPU %d '
                         '%s %s %s %s ======' %
                         (os.environ['CUDA_VISIBLE_DEVICES'], os.getpid(),
                          config.P, config.D, config.S, config.R))

        # cuda backend
        torch.backends.cudnn.benchmark = True
        self.cudaDeviceForAll = 'cuda:0'

        # meta params (not influencing model training)
        self.printFreq = 10
        self.tbxFreq = 500
        self.minDumpFreq = 500
        self.samplingVerbose = False
        self.monitorTrainFreq = 1000
        self.monitorValFreq = 1000

        # runnerComplementTheConfig
        self.config = self._runnerComplementTheConfig(config)

    @staticmethod
    def _runnerComplementTheConfig(config, **kwargs):
        datasetConfList = config.datasetConfList
        for datasetConf in datasetConfList:
            keys = list(datasetConf.keys())
            for k in keys:
                if k.startswith('batchSize'):
                    datasetConf[k + 'PerProcess'] = datasetConf[k]
        config.datasetMetaConf['batchSizeTotPerProcess'] = config.datasetMetaConf['batchSizeTot']

        return config

    def getTestTimeConfig(self):  # typically set all batchSize into 1
        testConf = copy.deepcopy(self.config)
        testConf.datasetMetaConf['batchSizeTotPerProcess'] = 1
        testConf.datasetConfList = [testConf.datasetConfList[0]]
        testConf.datasetConfList[0]['batchSizePerProcess'] = 1
        return testConf

    def metaDataLoading(self, **kwargs):
        config = self.config
        self.logger.info('[Trainer] MetaDataLoading')

        datasetMetaConf = config.datasetMetaConf
        datasetConfList = config.datasetConfList

        self.datasets = [
            CorenetChoySingleRenderingDataset(
                datasetConf,
                datasetSplit=datasetConf['trainingSplit'],
                datasetMode='trainFast',
                projRoot=self.projRoot,
            )
            for datasetConf in datasetConfList
        ]

        bmin = min([datasetConf['batchSizePerProcess'] for datasetConf in datasetConfList])
        if bmin >= 4:
            num_workers = 4
        elif bmin == 2:
            num_workers = 2
        else:  # bmin == 1 or 3
            num_workers = 1
        for did in range(len(datasetConfList)):
            assert datasetConfList[did]['batchSizePerProcess'] % num_workers == 0
        self.dataloaderList = [torch.utils.data.DataLoader(
            self.datasets[did], batch_size=datasetConfList[did]['batchSizePerProcess'],
            num_workers=4, shuffle=True, drop_last=True,
        ) for did in range(len(datasetConfList))]

        self.dataloaderIterList = [iter(dataloader) for dataloader in self.dataloaderList]

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

    def _netRandomInitialization(self):
        # random initialization
        # (Different threads might result in different model weights. Only Rank 0 is used)
        self.logger.info('[Trainer] MetaModelLoading - _netRandomInitialization')
        pass

    def _netResumingInitialization(self, **kwargs):
        iter_label = kwargs['iter_label']
        self.logger.info('[Trainer] MetaModelLoading - _netResumingInitialization (iter_label = %s)' %
                         str(iter_label))
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
        if resumeIter == 'latest' or resumeIter >= 0:
            for k in self.models.keys():
                if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                    _, self.models[k] = load_network(
                        self.logDir, self.models[k], k, resumeIter, map_location='cuda:0')
        if resumeIter == 'latest':
            resumeIter = -1
        return resumeIter

    def _netFinetuningInitialization(self):
        self.logger.info('[Trainer] MetaModelLoading - _netFinetuningInitialization')
        pass

    def _optimizerSetups(self):
        config = self.config
        self.logger.info('[Trainer] MetaModelLoading - _optimizerSetups')

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
            self.hookModels = {}
            for k in self.models.keys():
                if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                    self.hookModels[k] = PyTorchForwardHook(
                        hook_type, self.models[k],
                        '%s%s%s%sI%d' % (config.P, config.D, config.S, config.R, resumeIter))

    def metaModelLoading(self, **kwargs):
        # kwargs
        iter_label = kwargs['iter_label']
        hook_type = kwargs['hook_type']

        self._netConstruct(f0=float(self.meta['f0']))

        for k in self.models.keys():
            if k != 'meta':
                self.models[k].train()

        self._netRandomInitialization()
        resumeIter = self._netResumingInitialization(iter_label=iter_label)
        self._netFinetuningInitialization()

        self._optimizerSetups()
        self._netHookSetups(hook_type=hook_type, resumeIter=resumeIter)

        return resumeIter

    def metaLossPrepare(self):
        pass

    def setupMetaVariables(self):
        config = self.config
        datasetMetaConf = config.datasetMetaConf

        meta = {}
        meta = self._prepare_corenet_camcube(meta, cudaDevice=self.cudaDeviceForAll)
        if 'nearsurfacemassRForNeighbouringRange' in datasetMetaConf.keys():
            meta = self._prepare_for_voxelizationMasking(
                meta, cudaDevice=self.cudaDeviceForAll, skinNeighbouringName='nearsurfacemass',
                rForNeighbouring=datasetMetaConf['nearsurfacemassRForNeighbouringRange'],
            )
        if 'nearsurfaceairRForNeighbouringRange' in datasetMetaConf.keys():
            meta = self._prepare_for_voxelizationMasking(
                meta, cudaDevice=self.cudaDeviceForAll, skinNeighbouringName='nearsurfaceair',
                rForNeighbouring=datasetMetaConf['nearsurfaceairRForNeighbouringRange'],
            )
        if 'nonsurfaceRForNeighbouringRange' in datasetMetaConf.keys():
            meta = self._prepare_for_voxelizationMasking(
                meta, cudaDevice=self.cudaDeviceForAll, skinNeighbouringName='nonsurface',
                rForNeighbouring=datasetMetaConf['nonsurfaceRForNeighbouringRange'],
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
        config = self.config
        P = config['P']
        D = config['D']
        S = config['S']
        R = config['R']
        DVisualizer = getConfigGlobal(P, D, S, R)['exportedClasses']['DVisualizer']
        self.visualizerTrain = DVisualizer(
            visualDir=self.logDir + 'monitorTrain/', cudaDevice=self.cudaDeviceForAll,
            ifTracked=True,
        )
        self.visualizerTrain.setVisualMeta()
        self.visualizerVal = DVisualizer(
            visualDir=self.logDir + 'monitorVal/', cudaDevice=self.cudaDeviceForAll,
            ifTracked=True,
        )
        self.visualizerVal.setVisualMeta()

    def initializeAll(self, **kwargs):
        # kwargs
        iter_label = kwargs['iter_label']
        hook_type = kwargs['hook_type']

        self.metaDataLoading()
        self.setupMetaVariables()
        self.resumeIter = self.metaModelLoading(iter_label=iter_label,
                                                hook_type=hook_type,
                                                ifTrain=True)
        self.metaLossPrepare()
        self.setupMonitor()

        self.logger.info('Initialization finished!')

    def saveModelSnapshot(self, **kwargs):
        iterCount = kwargs['iterCount']
        for k in self.models.keys():
            if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                save_network(self.logDir, self.models[k], k, str(iterCount))

    def saveBatchVis(self, batch_vis, **kwargs):
        iterCount = kwargs['iterCount']
        pSaveMat(self.logDir + 'dump/train_iter_%d.mat' % iterCount,
                 {k: batch_vis[k] for k in batch_vis.keys()
                  if not k.startswith('feat') and not k.startswith('fullsurface')})

    def stepMonitorTrain(self, batch_vis, **kwargs):
        iterCount = kwargs['iterCount']

        config = self.config
        P = config['P']
        D = config['D']
        S = config['S']
        R = config['R']
        visualizer = self.visualizerTrain

        iou_list = []
        # nonDDPmodels = self.getNonDDPmodels()
        models = self.models
        for visIndex in range(config.datasetMetaConf['batchSizeTotPerProcess']):
            bsv0 = visualizer.constructInitialBatchStepVis0(
                batch_vis, iterCount=iterCount, visIndex=visIndex, P=P, D=D, S=S, R=R,
                verboseGeneral=visualizer.verboseGeneralMonitor,
            )
            bsv0 = visualizer.mergeFromBatchVis(bsv0, batch_vis)
            bsv0 = visualizer.stepMonitor(
                bsv0, models=models, meta=self.meta, datasets=self.datasets,
                generalVoxelizationFunc=self._generalVoxelization,
                samplingFromNetwork1ToNetwork2Func=self._samplingFromNetwork1ToNetwork2,
                samplingFromNetwork2ToNetwork3Func=self._samplingFromNetwork2ToNetwork3,
            )
            if visIndex < 4:
                visualizer.htmlMonitor(
                    bsv0, datasets=self.datasets,
                )
                visualizer.dumpMonitor(
                    bsv0,
                )
            iou_list.append(bsv0['corenetCubeIou'])
        return {'iou': np.array(iou_list, dtype=np.float32).mean()}

    def stepMonitorVal(self, **kwargs):
        iterCount = kwargs['iterCount']

        config = self.config
        P = config['P']
        D = config['D']
        S = config['S']
        R = config['R']
        visualizer = self.visualizerVal

        models = self.models
        batch0_np_list = []
        for did in range(len(config.datasetConfList)):
            for j in range(config.datasetConfList[did]['batchSizePerProcess']):
                batch0_np_list.append(self.datasets[did].getOneNP(int(
                    determinedArbitraryPermutedVector2(self.datasets[did].indVal)[j])))
        batch_np = {k: np.stack([x[k] for x in batch0_np_list], 0)
                    for k in batch0_np_list[0].keys()}
        batch_thcpu = castAnything(batch_np, 'np2thcpu')
        batch_thgpu = castAnything(batch_np, 'np2thgpu')
        batch_thgpu = self.batchPreprocessingTHGPU(
            batch_thcpu, batch_thgpu, datasets=self.datasets, datasetMetaConf=config.datasetMetaConf)
        batch_vis = castAnything(batch_thgpu, 'thgpu2np')
        iou_list = []
        for visIndex in range(config.datasetMetaConf['batchSizeTotPerProcess']):
            bsv0 = visualizer.constructInitialBatchStepVis0(
                batch_vis, iterCount=iterCount, visIndex=visIndex, P=P, D=D, S=S, R=R,
                verboseGeneral=visualizer.verboseGeneralMonitor,
            )
            bsv0 = visualizer.mergeFromBatchVis(bsv0, batch_vis)
            bsv0 = visualizer.stepMonitor(
                bsv0, models=models, meta=self.meta, datasets=self.datasets,
                generalVoxelizationFunc=self._generalVoxelization,
                samplingFromNetwork1ToNetwork2Func=self._samplingFromNetwork1ToNetwork2,
                samplingFromNetwork2ToNetwork3Func=self._samplingFromNetwork2ToNetwork3,
            )
            if visIndex < 4:
                visualizer.htmlMonitor(
                    bsv0, datasets=self.datasets,
                )
                visualizer.dumpMonitor(
                    bsv0,
                )
            iou_list.append(bsv0['corenetCubeIou'])
        return {'iou': np.array(iou_list, dtype=np.float32).mean()}

    def stepBatch(self, **kwargs):
        tmp = []
        for did in range(len(self.dataloaderIterList)):
            try:
                batchDid_thcpu = next(self.dataloaderIterList[did])
            except:
                self.dataloaderIterList[did] = iter(self.dataloaderList[did])
                batchDid_thcpu = next(self.dataloaderIterList[did])
            tmp.append(batchDid_thcpu)
        batch_thcpu = {k: torch.cat([t[k] for t in tmp], 0) for k in tmp[0].keys()}
        batch_thgpu = castAnything(batch_thcpu, 'thcpu2thgpu', device=self.cudaDeviceForAll)
        return batch_thcpu, batch_thgpu

    @staticmethod
    def samplingBatchTHGPUversionNeNs(batch_thcpu, batch_thgpu, **kwargs):
        datasetMetaConf = kwargs['datasetMetaConf']
        cudaDevice = kwargs['cudaDevice']
        f0 = kwargs['f0']

        samplingMethodList = datasetMetaConf['samplingMethodList']  # assume the same for all
        for samplingMethod in samplingMethodList:  # use fullsurface to compute sdf!
            if samplingMethod in ['uniform', 'nearsurface', 'surface']:
                for k in ['xyzCam', 'occfloat', 'sdf', 'gradSdfCam']:
                    if 'pre%sPC%s' % (bt(samplingMethod), k) in batch_thgpu.keys():
                        batch_thgpu['sufficient%sPC%s' % (bt(samplingMethod), k)] = \
                            batch_thgpu['pre%sPC%s' % (bt(samplingMethod), k)]  # symbolic
                        del batch_thgpu['pre%sPC%s' % (bt(samplingMethod), k)]  # delete immediately
                if 'sufficient%sPCmaskfloat' % bt(samplingMethod) not in batch_thgpu.keys():
                    batch_thgpu['sufficient%sPCmaskfloat' % bt(samplingMethod)] = \
                        torch.ones_like(batch_thgpu['sufficient%sPCxyzCam' % bt(samplingMethod)][:, :, 0])
                pass  # already stored presumed

            elif samplingMethod in ['selfnonsurface']:
                assert 'fullsurface' in samplingMethodList
                # You need them to determine whether close-to-surface
                selfnonsurfacePCxyzCam = torch.rand(
                    datasetMetaConf['batchSizeTotPerProcess'],
                    datasetMetaConf['numSamplingSufficientSelfnonsurface'], 3,
                    dtype=torch.float32, device=cudaDevice) - 0.5
                selfnonsurfacePCxyzCam[:, :, 2] += float(0.5 + f0 / 2.)
                tmp = knn_points(
                    selfnonsurfacePCxyzCam,
                    batch_thgpu['preFullsurfacePCxyzCam'],
                    K=1, return_nn=False, return_sorted=False,
                )
                distance = tmp.dists[:, :, 0] ** 0.5
                selfnonsurfacePCmaskfloat = torch.ones(
                    datasetMetaConf['batchSizeTotPerProcess'],
                    datasetMetaConf['numSamplingSufficientSelfnonsurface'],
                    dtype=torch.float32, device=cudaDevice)
                selfnonsurfacePCmaskfloat[distance <= datasetMetaConf['selfnonsurfaceDeltaRange']] = -2

                batch_thgpu['sufficientSelfnonsurfacePCxyzCam'] = selfnonsurfacePCxyzCam
                batch_thgpu['sufficientSelfnonsurfacePCmaskfloat'] = selfnonsurfacePCmaskfloat
                del selfnonsurfacePCxyzCam, tmp, distance, selfnonsurfacePCmaskfloat

            elif samplingMethod in ['fullsurface']:
                pass  # Do Nothing
            else:
                raise NotImplementedError('Not Yet Implemented for sampling method: %s' % samplingMethod)

        # xyzCam and xyzCamPersp
        for samplingMethod in samplingMethodList:
            if samplingMethod in ['fullsurface']:
                continue
            xyzCam = batch_thgpu['sufficient%sPCxyzCam' % bt(samplingMethod)]
            xyzCamPersp = camSys2CamPerspSysTHGPU(
                xyzCam,
                batch_thgpu['focalLengthWidth'], batch_thgpu['focalLengthHeight'],
                batch_thgpu['winWidth'], batch_thgpu['winHeight'])
            maskfloat = batch_thgpu['sufficient%sPCmaskfloat' % bt(samplingMethod)]
            maskfloat[
                ((xyzCamPersp[:, :, 2] <= 0) |
                 (xyzCamPersp[:, :, 0] <= -1) | (xyzCamPersp[:, :, 0] >= 1) |
                 (xyzCamPersp[:, :, 1] <= -1) | (xyzCamPersp[:, :, 1] >= 1)) &
                (maskfloat == 1)
                ] = -1.
            batch_thgpu['sufficient%sPCxyzCam' % bt(samplingMethod)] = xyzCam
            batch_thgpu['sufficient%sPCxyzCamPersp' % bt(samplingMethod)] = xyzCamPersp
            batch_thgpu['sufficient%sPCmaskfloat' % bt(samplingMethod)] = maskfloat

            # from sufficient to final
            #   flagging
            tmp = batch_thgpu['sufficient%sPCmaskfloat' % bt(samplingMethod)] > 0
            #   checking
            if tmp.sum(1).min() < datasetMetaConf['numSamplingFinal%s' % bt(samplingMethod)]:
                print('    [Fatal Error] samplingSufficient is not sufficient. Please check!')
                import ipdb
                ipdb.set_trace()
                raise ValueError('You cannot proceed.')
            #   bubble up
            tmp = torch.argsort(tmp.int(), dim=1, descending=True)[
                  :, :datasetMetaConf['numSamplingFinal%s' % bt(samplingMethod)]]
            tmp3 = tmp[:, :, None].repeat(1, 1, 3)
            #   gather all
            for k in ['xyzCam', 'xyzCamPersp', 'gradSdfCam']:
                if 'sufficient%sPC%s' % (bt(samplingMethod), k) in batch_thgpu.keys():
                    batch_thgpu['final%sPC%s' % (bt(samplingMethod), k)] = \
                        torch.gather(
                            batch_thgpu['sufficient%sPC%s' % (bt(samplingMethod), k)], dim=1, index=tmp3)
            for k in ['occfloat', 'sdf']:
                if 'sufficient%sPC%s' % (bt(samplingMethod), k) in batch_thgpu.keys():
                    batch_thgpu['final%sPC%s' % (bt(samplingMethod), k)] = \
                        torch.gather(
                            batch_thgpu['sufficient%sPC%s' % (bt(samplingMethod), k)], dim=1, index=tmp)
            #   checking
            maskfloatFinal = torch.gather(
                batch_thgpu['sufficient%sPCmaskfloat' % bt(samplingMethod)], dim=1, index=tmp)
            if not (maskfloatFinal > 0).all():
                print('    [Fatal Bug] maskfloat is not all zero. Please Check!')
                import ipdb
                ipdb.set_trace()
                raise ValueError('You cannot proceed.')

            # deleting
            del xyzCam, xyzCamPersp, tmp, tmp3, maskfloatFinal

        return batch_thgpu

    @staticmethod
    def _generalVoxelization(faceVertZO0List, Ly, Lx, Lz, cudaDevice):
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
        fill_voxels.fill_inside_voxels_gpu(grids_thgpu, inplace=True)
        return grids_thgpu  # (BZYX) bisem: 1 for the occupied

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
    def samplingBatchTHGPU(cls, batch_thcpu, batch_thgpu, **kwargs):
        meta = kwargs['meta']
        datasets = kwargs['datasets']
        datasetMetaConf = kwargs['datasetMetaConf']

        batchSizeTotPerProcess = datasetMetaConf['batchSizeTotPerProcess']

        f0 = meta['f0']

        faceVertZO0List = []
        for j in range(batch_thcpu['index'].shape[0]):
            did = int(batch_thcpu['did'][j])
            index = int(batch_thcpu['index'][j])
            tmp1 = datasets[did].getRawMeshCamOurs({}, index)
            face0 = tmp1['face'].astype(np.int32)
            tmp = tmp1['vertCamOurs'].astype(np.float32)
            tmp[:, :2] += 0.5
            tmp[:, -1] -= f0 / 2.
            vertZO0 = tmp
            faceVertZO0List.append(vertInfo2faceVertInfoNP(vertZO0[None], face0[None])[0])

        corenetCamcubeLabelBisemZYX, corenetCamcubeLabelSkinZYX = cls._generalVoxelizationWithSkin(
            faceVertZO0List,
            128, 128, 128, cudaDevice=batch_thgpu['index'].device)

        if 'samplingMethodList' not in datasetMetaConf.keys():
            samplingMethodList = ['gridpoint']
        else:
            samplingMethodList = datasetMetaConf['samplingMethodList']
        for samplingMethod in samplingMethodList:
            if samplingMethod == 'gridpoint':
                numSamplingFinalGridpoint = datasetMetaConf['numSamplingFinalGridpoint']
                ra_gridpoint = torch.randperm(
                    128 ** 3, device=batch_thgpu['index'].device)[:numSamplingFinalGridpoint]
                batch_thgpu['finalGridpointPCxyzCam'] = \
                    meta['corenetCamcubeVoxXyzCamZYX_thgpuDict'][128].reshape(
                        1, -1, 3)[:, ra_gridpoint, :].repeat(batchSizeTotPerProcess, 1, 1).detach()
                batch_thgpu['finalGridpointPClabelBisem'] = corenetCamcubeLabelBisemZYX.reshape(
                    batchSizeTotPerProcess, 128 ** 3)[:, ra_gridpoint].detach()
                del numSamplingFinalGridpoint, ra_gridpoint

            elif samplingMethod in ['nearsurfacemass', 'nearsurfaceair', 'nonsurface']:
                numSampling = datasetMetaConf['numSamplingFinal%s' % bt(samplingMethod)]
                kernelForNeighbouring = meta['%s_kernelForNeighbouring_thgpu' % samplingMethod]
                rForNeighbouring = meta['%s_rForNeighbouring' % samplingMethod]
                gridPurtFactor = datasetMetaConf['%sGridPurtFactor' % samplingMethod]
                corenetCamcubeLabelSkinNeighbouringZYX = (F.conv3d(
                    input=corenetCamcubeLabelSkinZYX[:, None, :, :, :],
                    weight=kernelForNeighbouring, bias=None, stride=1,
                    padding=rForNeighbouring,
                )[:, 0, :, :, :] > 0).float()
                # flagging
                if samplingMethod == 'nearsurfacemass':
                    valid = (corenetCamcubeLabelSkinNeighbouringZYX *
                             corenetCamcubeLabelBisemZYX).reshape(batchSizeTotPerProcess, -1)
                elif samplingMethod == 'nearsurfaceair':
                    valid = (corenetCamcubeLabelSkinNeighbouringZYX *
                             (1 - corenetCamcubeLabelBisemZYX)).reshape(batchSizeTotPerProcess, -1)
                elif samplingMethod == 'nonsurface':
                    valid = (1 - corenetCamcubeLabelSkinNeighbouringZYX).reshape(batchSizeTotPerProcess, -1)
                else:
                    raise NotImplementedError('Unknown samplingMethod: %s' % samplingMethod)
                batch_thgpu['%sMaskRate' % samplingMethod] = valid.float().sum(1) / valid.shape[1]
                # bubble up
                tmp = torch.argsort(
                    valid * (1 + torch.rand(  # 1+ is important, as sometimes your rand returns 0.
                        batchSizeTotPerProcess, valid.shape[1],
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
                        1, -1, 3).repeat(batchSizeTotPerProcess, 1, 1),
                    dim=1, index=tmp3)
                pcXyzCam += (2. * torch.rand(
                    pcXyzCam.shape[0], pcXyzCam.shape[1], pcXyzCam.shape[2],
                    dtype=torch.float32, device=pcXyzCam.device) - 1.) * \
                    (gridPurtFactor * 0.5 / 128)
                #    verification purpose
                pcLabelBisem = torch.gather(
                    corenetCamcubeLabelBisemZYX.view(batchSizeTotPerProcess, -1), dim=1, index=tmp)
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
                batch_thgpu['final%sPCxyzCam' % bt(samplingMethod)] = pcXyzCam
                batch_thgpu['final%sPClabelBisem' % bt(samplingMethod)] = pcLabelBisem
            else:
                raise NotImplementedError('Unknown samplingMethod: %s' % samplingMethod)

        batch_thgpu['corenetCamcubeLabelBisemZYX'] = corenetCamcubeLabelBisemZYX

        return batch_thgpu

    def batchPreprocessingTHGPU(self, batch_thcpu, batch_thgpu, **kwargs):
        datasets = kwargs['datasets']

        config = self.config
        datasetMetaConf = config.datasetMetaConf

        with torch.no_grad():
            batch_thgpu = self.samplingBatchTHGPU(
                batch_thcpu,
                batch_thgpu,
                meta=self.meta,
                datasets=datasets,
                datasetMetaConf=datasetMetaConf,
                L=datasetMetaConf.get('gridPointLength', None),
            )
        return batch_thgpu

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
    def forwardNet(cls, batch_thgpu, **kwargs):
        models = kwargs['models']
        meta = kwargs['meta']

        corenetCamcubeVoxXyzCamPerspZYX_thgpuDict = \
            meta['corenetCamcubeVoxXyzCamPerspZYX_thgpuDict']

        encoder = models['encoder']
        networkTwo = models['networkTwo']

        # network 1
        image_initial_thgpu = batch_thgpu['img']
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
        batch_thgpu['corenetCamcubePredBisemZYX'] = yy['y6'][:, 0, :, :, :]

        for samplingMethod in ['gridpoint']:
            pCam = batch_thgpu['final%sPCxyzCam' % bt(samplingMethod)]

            zz = cls._samplingFromNetwork2ToNetwork3(
                yy, pCam,
                f0=meta['f0'],
            )

            batch_thgpu['final%sPCpredBisem' % bt(samplingMethod)] = \
                zz['z6'].view(pCam.shape[0], pCam.shape[1])
        return batch_thgpu

    @classmethod
    def forwardLoss(cls, batch_thgpu, **kwargs):
        wl = kwargs['wl']

        # lossGridpointIou
        if 'lossGridpointIou' in wl.keys() and wl['lossGridpointIou'] > 0:
            pred = batch_thgpu['finalGridpointPCpredBisem']
            label = batch_thgpu['finalGridpointPClabelBisem']
            B = label.shape[0]
            iou = torch.div(
                torch.min(pred, label).reshape(B, -1).sum(1),
                torch.max(pred, label).reshape(B, -1).sum(1) + 1.e-4,
            )
            batch_thgpu['lossGridpointIou'] = (1. - iou).mean()
            del pred, label, B, iou

        # statIou
        pred = batch_thgpu['corenetCamcubePredBisemZYX'] > 0.5
        label = batch_thgpu['corenetCamcubeLabelBisemZYX']
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

        config = self.config

        batch_thgpu = self.forwardNet(
            batch_thgpu,
            models=self.models,
            perturbDelta=config.get('perturbDelta', None),
            meta=self.meta,
            uuUsedKeys=config.get('uuUsedKeys', None),
            zzUsedKeys=config.get('zzUsedKeys', None),
            L=config.datasetMetaConf.get('gridPointLength', None),
            c_dim_specs=config.get('c_dim_specs', None),
        )
        batch_thgpu = self.forwardLoss(
            batch_thgpu,
            wl=config.wl,
            iterCount=iterCount,
            logDir=self.logDir,
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

    def trainNoException(self):
        config = self.config
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
            power10 = min(power10, 5)  # to dump more models - the small batch size 4 would oscillate the performance significantly, so evaluating more models would help.
            divider = 10 ** power10

            ifStore = iterCount > self.resumeIter and (
                    iterCount % max([divider, self.minDumpFreq]) == 0
                    or iterCount == self.resumeIter + 1)
            ifBackupTheLatestModels = (iterCount % 10000 == 0) and (iterCount > 100000)
            monitorMode = 3
            ifMonitorTrain = (iterCount % self.monitorTrainFreq == monitorMode) or \
                             (iterCount < self.monitorTrainFreq and iterCount % (
                                         self.monitorTrainFreq / 10) == monitorMode)
            ifMonitorVal = (iterCount % self.monitorValFreq == monitorMode) or \
                           (iterCount < self.monitorValFreq and iterCount % (self.monitorValFreq / 10) == monitorMode)

            ifPrint = ifStore or (iterCount - self.resumeIter) % self.printFreq == 0 or iterCount < 20
            if ifMonitorTrain or ifMonitorVal:
                ifPrint = True
            # Note ifStore do not do this - all the threads needs sync. ifPrint does not relate to sync.
            ifSaveToFinishedIter = iterCount % 50 == 0

            # ifTracked
            ifTracked = True

            # printing
            if ifPrint and ifTracked:
                self.logger.info(
                    '---------- Iter %d Training: GPU %s CPU %d '
                    '%s %s %s %s ---------' %
                    (iterCount, os.environ['CUDA_VISIBLE_DEVICES'], os.getpid(),
                     config.P, config.D, config.S, config.R))

            # -------------------------------- batch data loading -------------------------------- #
            t = time.time()

            batch_thcpu, batch_thgpu = self.stepBatch()

            if ifPrint and ifTracked:
                self.logger.info('    [Timer] data loading is %.3f seconds' % (time.time() - t))

            # ----------------------------------- Preprocessing ----------------------------------- #
            t = time.time()
            batch_thgpu = self.batchPreprocessingTHGPU(batch_thcpu, batch_thgpu,
                                                       datasets=self.datasets,
                                                       datasetMetaConf=self.config.datasetMetaConf)
            if ifPrint and ifTracked:
                self.logger.info('    [Timer] batchPreprocessingTHGPU is %.3f seconds' % (time.time() - t))

            # ------------------------------------ main course ------------------------------------ #
            t = time.time()

            batch_thgpu = self.forwardBackwardUpdate(batch_thgpu,
                                                     ifTrain=True,
                                                     iterCount=iterCount,
                                                     datasetMetaConf=config.datasetMetaConf,
                                                     ifAllowBackward=True)
            if ifPrint and ifTracked:
                self.logger.info('    [Timer] forwardBackwardUpdate is %.3f seconds' % (time.time() - t))

            # ---------------------------------------- Meta --------------------------------------- #
            ifMonitorTrain = False
            ifMonitorVal = False
            if ifMonitorTrain:
                with torch.no_grad():
                    for k in self.models:
                        if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                            self.models[k].eval()
                    batch_vis = castAnything(batch_thgpu, 'thgpu2np')
                    tmp = self.stepMonitorTrain(
                        batch_vis, iterCount=iterCount)
                    batch_thgpu['statIouTrain'] = \
                        torch.tensor(tmp['iou']).float().to(self.cudaDeviceForAll)
                    for k in self.models:
                        if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                            self.models[k].train()

            if ifMonitorVal:
                with torch.no_grad():
                    for k in self.models:
                        if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                            self.models[k].eval()
                    tmp = self.stepMonitorVal(
                        iterCount=iterCount,
                    )
                    batch_thgpu['statIouVal'] = \
                        torch.tensor(tmp['iou']).float().to(self.cudaDeviceForAll)
                    for k in self.models:
                        if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                            self.models[k].train()

            if ifPrint and ifTracked:
                for k in batch_thgpu.keys():
                    if k.startswith('loss'):
                        self.logger.info('    [Loss] %s Iter%d: %.5f' %
                                         (k, iterCount, float(batch_thgpu[k])))
                    if k.startswith('stat'):
                        self.logger.info('    [Stat] %s Iter%d: %.5f' %
                                         (k, iterCount, float(batch_thgpu[k])))

            if ifSaveToFinishedIter and ifTracked:
                self.finishedIterCount = iterCount
                self.finishedBatchVis = castAnything(batch_thgpu, 'thgpu2np')

            if ifStore and ifTracked:
                # save model snapshot
                self.saveModelSnapshot(iterCount=iterCount)

                # visualTrain direct saving (not including visualization)
                batch_vis = castAnything(batch_thgpu, 'thgpu2np')
                self.saveBatchVis(batch_vis, iterCount=iterCount)
                del batch_thgpu
            else:
                del batch_thgpu

            if ifBackupTheLatestModels and ifTracked:
                self.saveModelSnapshot(iterCount='latest')

            if ifPrint and ifTracked:
                self.logger.info('[Timer] Total is %.3f seconds.' % (time.time() - timeIterStart))

            iterCount += 1
            pass

    def train(self):
        try:
            self.trainNoException()
        except KeyboardInterrupt:
            self.logger.info('KeyboardInterrupt Detected')
            if self.finishedIterCount > 10000:
                self.saveModelSnapshot(iterCount=self.finishedIterCount)
                self.saveBatchVis(self.finishedBatchVis, iterCount=self.finishedIterCount)


