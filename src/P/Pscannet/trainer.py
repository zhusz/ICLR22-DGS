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

# add python path
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/' + 'lib/')

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
from codes_py.np_ext.data_processing_utils_v1 import determinedArbitraryPermutedVector2correct
from codes_py.toolbox_3D.self_sampling_v1 import mesh_sampling_given_normal_np_simple
from codes_py.toolbox_framework.framework_util_v2 import constructInitialBatchStepVis0, \
    mergeFromBatchVis
from codes_py.toolbox_3D.pyrender_wrapper_v1 import PyrenderManager
from codes_py.toolbox_3D.dgs_wrapper_v1 import DGSLayer

# PDSR
from .dataset import PScannetGivenRenderDataset
from .lib.multi_depth_model_woauxi import DepthModel
from .resnetfc import ResNetFC
from .positionalEncoding import PositionalEncoding

# visualizer from configs_registration
from configs_registration import getConfigGlobal
from datasets_registration import datasetRetrieveList

# benchmarking
from Bpscannet.testDataEntry.scannetGivenRenderDataset import ScannetMeshCache
from Bpscannet.testDataEntry.testDataEntryPool import getTestDataEntryDict
from Bpscannet.csvGeometry3DEntry.benchmarkingGeometry3DScannet \
    import benchmarkingGeometry3DScannetFunc

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
        self.monitorValFreq = 40000
        self.monitorDumpFreq = 200000
        self.monitorMode = 3  # which iteration is the first to step the monitors

        # runnerComplementTheConfig
        self.config = self._runnerComplementTheConfig(config, numMpProcess=self.numMpProcess)

    @staticmethod
    def _runnerComplementTheConfig(config, **kwargs):
        # This function cannot be called during demo time. Demo does not allow multiple GPU.
        numMpProcess = kwargs['numMpProcess']
        datasetConfList = config.datasetConfList
        for datasetConf in datasetConfList:
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
        if numMpProcess > 0:
            config.datasetMetaConf['batchSizeTotPerProcess'] = int(
                config.datasetMetaConf['batchSizeTot'] / numMpProcess
            )
        else:
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
            PScannetGivenRenderDataset(
                datasetConf,
                datasetSplit=datasetConf['trainingSplit'],
                datasetMode='trainFast',
                projRoot=self.projRoot,
                R=config.R,
                ifDebugMode=False,
            )
            for datasetConf in datasetConfList
        ]

        batchSizeTotPerProcess = int(datasetMetaConf['batchSizeTotPerProcess'])
        num_workers = int(math.ceil(float(batchSizeTotPerProcess) / 8.))
        self.dataloaderList = [torch.utils.data.DataLoader(
            self.datasets[did], batch_size=datasetConfList[did]['batchSizePerProcess'],
            num_workers=num_workers, shuffle=True, drop_last=True,
        ) for did in range(len(datasetConfList))]

        self.dataloaderIterList = [iter(dataloader) for dataloader in self.dataloaderList]

    def _netConstruct(self, **kwargs):
        config = self.config
        self.logger.info('[Trainer] MetaModelLoading - _netConstruct')
        projRoot = self.projRoot

        encoder = DepthModel(encoder=config.depthEncoderTag)

        if config.ifFinetunedFromAdelai == 0:
            pass  # using the default initialization
        elif config.ifFinetunedFromAdelai == 1:
            if config.depthEncoderTag == 'resnet50_stride32':
                adelai_pretrained_weight_file = \
                    projRoot + 'remote_syncdata/ICLR22-DGS/v_external_codes/AdelaiDepth/LeRes/res50.pth'
            elif config.depthEncoderTag == 'resnext101_stride32x8d':
                adelai_pretrained_weight_file = \
                    projRoot + 'remote_syncdata/ICLR22-DGS/v_external_codes/AdelaiDepth/LeRes/res101.pth'
            else:
                raise NotImplementedError('Unknown config.adelaiHalfEncoderTag: %s' %
                                          config.adelaiHalfEncoderTag)
            sd = torch.load(
                adelai_pretrained_weight_file, map_location='cpu') \
                ['depth_model']
            sd = {k[19:]: sd[k] for k in sd.keys()}
            encoder.load_state_dict(sd)
        else:
            raise NotImplementedError('Unknown ifFinetunedFromAdelai: %d' % config.ifFinetunedFromAdelai)

        c_dim = sum(list(config.c_dim_specs.values()))
        decoder = ResNetFC(d_in=39, d_out=1, n_blocks=5, d_latent=c_dim, d_hidden=512, beta=0.0,
                           combine_layer=3)
        positionalEncoder = PositionalEncoding(
            num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True,
        )
        dgsLayer = DGSLayer(fw=config.datasetMetaConf['fScaleWidth'],
                            fh=config.datasetMetaConf['fScaleHeight'])
        meta = {
            'nonLearningModelNameList': ['positionalEncoder', 'dgsLayer'],
            'c_dim_specs': config.c_dim_specs,
        }
        self.models = dict()
        self.models['decoder'] = decoder.to(self.cudaDeviceForAll)
        self.models['encoder'] = encoder.to(self.cudaDeviceForAll)
        self.models['positionalEncoder'] = positionalEncoder.to(self.cudaDeviceForAll)
        self.models['dgsLayer'] = dgsLayer.to(self.cudaDeviceForAll)
        self.models['meta'] = meta

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

        self._netConstruct()

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
        meta = {}

        meta = self._meta_initialize_scannet_house_vert_world_cache(meta)
        self.meta = meta

    @staticmethod
    def _meta_initialize_scannet_house_vert_world_cache(meta):
        scannetMeshCache = ScannetMeshCache()
        meta['scannetMeshCache'] = scannetMeshCache
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
            ifTracked=self.rank == self.trackedRank,
        )
        self.visualizerTrain.setVisualMeta(voxSize=128)
        self.visualizerVal = DVisualizer(
            visualDir=self.logDir + 'monitorVal/', cudaDevice=self.cudaDeviceForAll,
            ifTracked=self.rank == self.trackedRank,
        )
        self.visualizerVal.setVisualMeta(voxSize=128)
        self.testDataEntry = getTestDataEntryDict(wishedTestDataNickName=['scannetOfficialTestSplit10'])['scannetOfficialTestSplit10']

    def initializeAll(self, **kwargs):
        # kwargs
        iter_label = kwargs['iter_label']  # required for all cases
        hook_type = kwargs['hook_type']  # required for all cases
        if_need_metaDataLoading = kwargs['if_need_metaDataLoading']

        if if_need_metaDataLoading:
            self.metaDataLoading()
        self.resumeIter = self.metaModelLoading(iter_label=iter_label,
                                                hook_type=hook_type)
        self.metaLossPrepare()
        if if_need_metaDataLoading:
            self.setupMetaVariables()
        else:
            self.meta = {}
        if if_need_metaDataLoading:
            self.setupMonitor()

        self.logger.info('Initialization finished! Rank = %d' % self.rank)

    def saveModelSnapshot(self, **kwargs):
        iterCount = kwargs['iterCount']
        if self.rank == self.trackedRank:
            if self.numMpProcess > 0:  # self.net.module
                for k in self.models.keys():
                    if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                        save_network(self.logDir, self.models[k].module, k, str(iterCount))
            else:  # self.net
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

        config = self.config
        P = config['P']
        D = config['D']
        S = config['S']
        R = config['R']
        visualizer = self.visualizerTrain

        pcList = ['evalPred']
        itemList = ['acc', 'compl', 'chamfer', 'prec', 'recall', 'F1']
        metricOrFitList = ['metric', 'fit']
        viaDOrNotList = ['']
        keys = []
        for pc in pcList:
            for item in itemList:
                for metricOrFit in metricOrFitList:
                    for viaDOrNot in viaDOrNotList:
                        keys.append('%sPCBen%s%s%s' %
                                    (pc, bt(metricOrFit), bt(item), viaDOrNot))
        depthBenKeyList = ['depthRenderedFromEvalPred2']
        itemList = ['r1', 'AbsRel', 'SqRel', 'RMSE']
        metricOrFitList = ['metric', 'fit']
        viaDOrNotList = ['']
        for depth in depthBenKeyList:
            for item in itemList:
                for metricOrFit in metricOrFitList:
                    for viaDOrNot in viaDOrNotList:
                        keys.append('%sDBen%s%s%s' %
                                    (depth, bt(metricOrFit), bt(item), viaDOrNot))
        ben = {k: [] for k in keys}
        models = self.models
        for visIndex in range(min(8, config.datasetMetaConf['batchSizeTotPerProcess'])):
            bsv0 = constructInitialBatchStepVis0(
                batch_vis, iterCount=iterCount, visIndex=visIndex, P=P, D=D, S=S, R=R,
                verboseGeneral=visualizer.verboseGeneralMonitor,
            )
            bsv0 = mergeFromBatchVis(bsv0, batch_vis)
            ifRequiresDrawing = (visIndex < 4) and ifIsTrackedRank
            bsv0 = visualizer.stepMonitorTrain(
                bsv0, models=models, meta=self.meta, datasets=self.datasets,
                Trainer=type(self), ifRequiresDrawing=ifRequiresDrawing,
                ifRequiresPredictingHere=True,
            )
            if ifRequiresDrawing:
                visualizer.htmlMonitor(
                    bsv0, numMpProcess=self.numMpProcess,
                )
                if ifMonitorDump:
                    visualizer.dumpMonitor(
                        bsv0,
                    )
            for k in keys:
                if k in bsv0.keys():
                    ben[k].append(float(bsv0[k]))
        ben = {k: np.array(ben[k], dtype=np.float32) for k in ben.keys()}
        return ben

    def stepMonitorVal(self, **kwargs):
        iterCount = kwargs['iterCount']
        ifMonitorDump = kwargs['ifMonitorDump']
        ifIsTrackedRank = self.rank == self.trackedRank

        config = self.config
        P = config['P']
        D = config['D']
        S = config['S']
        R = config['R']
        visualizer = self.visualizerVal

        testDataEntry = self.testDataEntry
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

        pyrenderManager = visualizer.pyrenderManager

        if numMpProcess <= 0:
            j1 = 0
            j2 = j_tot
        else:
            j1 = int(math.floor(float(j_tot) / realNumMpProcess * rank))
            j2 = int(math.floor(float(j_tot) / realNumMpProcess * (rank + 1)))
        testDataNickName = testDataEntry['testDataNickName']
        benRecord = {}
        # j2 = j1 + 1  # debug
        keys_3D = []
        itemList = ['acc', 'compl', 'chamfer', 'prec', 'recall', 'F1']
        metricOrFitList = ['metric', 'fit']
        for item in itemList:
            for metricOrFit in metricOrFitList:
                keys_3D.append('%s%s' % (metricOrFit, bt(item)))
        keys_2D = []
        itemList = ['r1', 'AbsRel', 'SqRel', 'RMSE', 'AbsDiff', 'r2', 'r3', 'complete', 'LogRMSE']
        metricOrFitList = ['metric', 'fit']
        for item in itemList:
            for metricOrFit in metricOrFitList:
                keys_2D.append('%s%s' % (metricOrFit, bt(item)))
        for j in range(j1, j2):
            index = indChosen[j]
            print('[Trainer Visualizer Val] Progress Iter%d: '
                  'testDataNickName %s, index = %d, j = %d, j1 = %d, j2 = %d, '
                  'rank = %d, numMpProcess = %d, j_tot = %d' %
                  (iterCount, testDataNickName, index, j, j1, j2,
                   rank, numMpProcess, j_tot))
            batch0_np = testDataEntry['datasetObj'].getOneNP(index)
            batch_np = {k: batch0_np[k][None] for k in batch0_np.keys()}
            batch_vis = batch_np
            bsv0_initial = constructInitialBatchStepVis0(
                batch_vis, iterCount=iterCount, visIndex=0,
                P=P, D=D, S=S, R=R,
                verboseGeneral=0,
            )
            bsv0_initial = mergeFromBatchVis(bsv0_initial, batch_vis)
            # step
            try:
                if testDataNickName in ['scannetOfficialTestSplit10']:
                    bsv0 = benchmarkingGeometry3DScannetFunc(
                        bsv0_initial,
                        scannetMeshCache=self.meta['scannetMeshCache'],
                        datasets=[testDataEntry['datasetObj']],
                        # misc
                        cudaDevice=self.cudaDeviceForAll,
                        raiseOrTrace='raise',
                        # benchmarking rules
                        voxSize=128, numMeshSamplingPoint=200000,
                        # drawing
                        ifRequiresDrawing=(index in indVisChosen) and (rank == trackedRank),
                        pyrenderManager=pyrenderManager,
                        # predicting
                        ifRequiresPredictingHere=True,
                        models=self.models, verboseBatchForwarding=0,
                        Trainer=type(self), doPred0Func=visualizer.doPred0,
                        useSurfaceOrDepth='surface', meta=self.meta,
                    )
                elif testDataNickName in ['mpv1rebuttal']:
                    bsv0 = benchmarkingMpv1SingleViewReconstruction(
                        bsv0_initial,
                        mpv1MeshCache=self.meta['mpv1MeshCache'],
                        datasets=[testDataEntry['datasetObj']],
                        # misc
                        cudaDevice=self.cudaDeviceForAll,
                        raiseOrTrace='raise',
                        # benchmarking rules
                        voxSize=128, numMeshSamplingPoint=200000,
                        # drawing
                        ifRequiresDrawing=(index in indVisChosen) and (rank == trackedRank),
                        pyrenderManager=pyrenderManager,
                        # predicting
                        ifRequiresPredictingHere=True,
                        models=self.models, verboseBatchForwarding=0,
                        Trainer=type(self), doPred0Func=visualizer.doPred0,
                        useSurfaceOrDepth='surface', meta=self.meta,
                    )
                elif testDataNickName in ['demo1', 'freedemo1']:
                    bsv0 = benchmarkingDemoSingleViewReconstruction(
                        copy.deepcopy(bsv0_initial),
                        datasets=[testDataEntry['datasetObj']],
                        # misc
                        cudaDevice=self.cudaDeviceForAll,
                        raiseOrTrace='raise',
                        # benchmarking rules
                        voxSize=256, numMeshSamplingPoint=200000,
                        # drawing
                        ifRequiresDrawing=(index in indVisChosen) and (rank == trackedRank),
                        pyrenderManager=pyrenderManager,
                        # predicting
                        ifRequiresPredictingHere=True,
                        models=self.models, verboseBatchForwarding=0,
                        Trainer=type(self), doPred0Func=visualizer.doPred0,
                        useSurfaceOrDepth='surface', meta=self.meta,
                    )
                else:
                    raise NotImplementedError('Unknown testDataNickName: %s' % testDataNickName)

                # final benchmarkings - finalBen
                if testDataNickName in ['scannetOfficialTestSplit10', 'mpv1rebuttal']:
                    for k in keys_3D:
                        bsv0['finalBen%s' % bt(k)] = bsv0['evalPredPCBen%s' % bt(k)]
                    for k in keys_2D:
                        bsv0['finalBen%s' % bt(k)] = \
                            bsv0['depthRenderedFromEvalPred2DBen%s' % bt(k)]
                elif testDataNickName in ['freedemo1', 'demo1']:
                    pass
                else:
                    raise NotImplementedError('Unknown testDataNickName: %s' % testDataNickName)

                bsv0_toStore = {}
                for k in bsv0.keys():
                    if k.startswith('finalBen') or k in [
                        'iterCount', 'visIndex', 'P', 'D', 'S', 'R', 'methodologyName',
                        'index', 'did', 'datasetID', 'dataset', 'flagSplit',
                        'houseID', 'viewID',
                    ]:
                        bsv0_toStore[k] = bsv0[k]
                benRecord[j] = bsv0_toStore

                if (index in indVisChosen) and (rank == trackedRank):
                    visualizer.htmlMonitor(
                        bsv0, numMpProcess=self.numMpProcess,
                    )
                    if ifMonitorDump:
                        visualizer.dumpMonitor(
                            bsv0
                        )
            except Exception:
                pass

        # ben = {k: np.array(ben[k], dtype=np.float32) for k in ben.keys()}
        ar = []
        for br in benRecord.values():
            ar.append(np.array([
                br['finalBen%s' % bt(k)] for k in keys_2D + keys_3D
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
        assert validCount > 0,  'validCount is 0'
        arMean = arSum_thgpu.detach().cpu().numpy() / float(validCount)
        keys = keys_2D + keys_3D
        ben = {'finalBen%s' % bt(keys[j]): np.array(arMean[j], dtype=np.float32)
               for j in range(len(keys))}
        ben['finalBenValidCount'] = np.array(validCount, dtype=np.float32)
        return ben

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

    @classmethod
    def samplingBatchTHGPU(cls, batch_thcpu, batch_thgpu, **kwargs):
        datasets = kwargs['datasets']
        datasetMetaConf = kwargs['datasetMetaConf']
        meta = kwargs['meta']
        projRoot = kwargs['projRoot']
        cudaDevice = kwargs['cudaDevice']
        r = kwargs['r']  # orthogonal volume space
        iterCount = kwargs['iterCount']

        samplingMethodList = datasetMetaConf['samplingMethodList']
        B = int(batch_thcpu['index'].shape[0])

        # cache load mesh and sample face Centroid
        finalFullsurfacePCxyzWorld = np.zeros((
            B, datasetMetaConf['numSamplingFinalFullsurface'], 3), dtype=np.float32)
        finalFullsurfacePCnormalWorld = np.zeros((
            B, datasetMetaConf['numSamplingFinalFullsurface'], 3), dtype=np.float32)
        # finalFullsurfacePCnyu40ID = np.zeros((
        #     B, datasetMetaConf['numSamplingFinalFullsurface'], ), dtype=np.int32)
        finalFullsurfacePCpix3d10ID = np.zeros((
            B, datasetMetaConf['numSamplingFinalFullsurface'], ), dtype=np.int32)
        minBoundWorld = np.zeros((B, 3), dtype=np.float32)
        maxBoundWorld = np.zeros((B, 3), dtype=np.float32)
        for j in range(B):
            index0 = int(batch_thcpu['index'][j])
            houseID0 = int(batch_thcpu['houseID'][j])
            viewID0 = int(batch_thcpu['viewID'][j])
            did = int(batch_thcpu['did'][j])
            datasetID = int(batch_thcpu['datasetID'][j])

            # cache loading
            tmp0 = meta['scannetMeshCache'].call_cache_scannet_house_vert_world_0(
                houseID0=houseID0,
                scannetFile=datasets[did].fileList_house[houseID0],
                scannetScanID=datasets[did].scanIDList_house[houseID0],
                verbose=(iterCount >= 100),
                original_dataset_root=datasets[did].original_dataset_root,
            )
            vertWorld0 = tmp0['vertWorld0']
            face0 = tmp0['face0']
            faceNormalWorld0 = tmp0['faceNormalWorld0']
            packedFaceFlag0 = udl('mats_R17fsw%.2ffsh%.2f_packedFaceFlag0' %
                                  (datasetMetaConf['fScaleWidth'],
                                   datasetMetaConf['fScaleHeight']),
                                  datasetRetrieveList[datasetID], index0)
            faceFlag0 = np.unpackbits(packedFaceFlag0, bitorder='big')[:face0.shape[0]].astype(bool)
            face0 = face0[faceFlag0]
            faceNormalWorld0 = faceNormalWorld0[faceFlag0]
            faceVertWorld0 = vertInfo2faceVertInfoNP(vertWorld0[None], face0[None])[0]
            tmp1 = mesh_sampling_given_normal_np_simple(
                datasetMetaConf['numSamplingFinalFullsurface'], faceVertWorld0)
            finalFullsurfacePCxyzWorld[j, :, :] = tmp1['point0']
            finalFullsurfacePCnormalWorld[j, :, :] = faceNormalWorld0[tmp1['pointFace0'], :]
            minBoundWorld[j, :] = tmp1['point0'].min(0)
            maxBoundWorld[j, :] = tmp1['point0'].max(0)
        minBoundWorld_thgpu = torch.from_numpy(minBoundWorld).to(cudaDevice)
        maxBoundWorld_thgpu = torch.from_numpy(maxBoundWorld).to(cudaDevice)

        if 'fullsurface' in samplingMethodList:
            batch_thgpu['finalFullsurfacePCxyzWorld'] = \
                torch.from_numpy(finalFullsurfacePCxyzWorld).to(cudaDevice)
            batch_thgpu['finalFullsurfacePCgradSdfWorld'] = \
                torch.from_numpy(finalFullsurfacePCnormalWorld).to(cudaDevice)
            batch_thgpu['finalFullsurfacePCsemantic'] = \
                torch.from_numpy(finalFullsurfacePCpix3d10ID).to(cudaDevice)

        if 'surface' in samplingMethodList:
            batch_thgpu['sufficientSurfacePCxyzWorld'] = \
                batch_thgpu['finalFullsurfacePCxyzWorld'][
                :, -datasetMetaConf['numSamplingSufficientSurface']:, :]
            batch_thgpu['sufficientSurfacePCgradSdfWorld'] = \
                batch_thgpu['finalFullsurfacePCgradSdfWorld'][
                :, -datasetMetaConf['numSamplingSufficientSurface']:, :]
            batch_thgpu['sufficientSurfacePCmaskfloat'] = torch.ones(
                B, datasetMetaConf['numSamplingSufficientSurface'],
                dtype=torch.float32, device=cudaDevice)

        if 'nearsurface' in samplingMethodList:
            d = torch.rand(B, datasetMetaConf['numSamplingSufficientNearsurface'], device=cudaDevice)
            d = (2 * d - 1) * datasetMetaConf['nearsurfaceDeltaRange']
            batch_thgpu['sufficientNearsurfacePCxyzWorld'] = \
                batch_thgpu['finalFullsurfacePCxyzWorld'][
                :, :datasetMetaConf['numSamplingSufficientNearsurface']] + \
                d[:, :, None] * batch_thgpu['finalFullsurfacePCgradSdfWorld'][
                                :, :datasetMetaConf['numSamplingSufficientNearsurface']]
            batch_thgpu['sufficientNearsurfacePCoccfloat'] = (d > 0).float()
            batch_thgpu['sufficientNearsurfacePCsemantic'] = (
                (batch_thgpu['finalFullsurfacePCsemantic'][
                    :, :datasetMetaConf['numSamplingSufficientNearsurface']
                ])
                * (d <= 0)
            ).int()

            batch_thgpu['sufficientNearsurfacePCmaskfloat'] = torch.ones_like(d)
            # Note we still need those semantic unknown point
            #  - just that during loss imposition for semanitcs, we need to mask them out
            #    but for loss imposition for occupancies, we still need them
            #    Hence we use the above one, not the following one
            # batch_thgpu['sufficientNearsurfacePCmaskfloat'] = \
            #     (batch_thgpu['sufficientNearsurfacePCnyu40ID'] >= 0).float()

            batch_thgpu['sufficientNearsurfacePCsdf'] = d

        if 'selfnonsurface' in samplingMethodList:
            t = torch.rand(
                B, datasetMetaConf['numSamplingSufficientSelfnonsurface'], 3, device=cudaDevice)
            t = t * (maxBoundWorld_thgpu - minBoundWorld_thgpu)[:, None, :] + \
                minBoundWorld_thgpu[:, None, :]
            tmp = knn_points(
                t, batch_thgpu['finalFullsurfacePCxyzWorld'],
                K=1, return_nn=False, return_sorted=False,
            )
            dists = tmp.dists[:, :, 0] ** 0.5
            batch_thgpu['sufficientSelfnonsurfacePCxyzWorld'] = t
            batch_thgpu['sufficientSelfnonsurfacePCmaskfloat'] = torch.ones_like(dists)
            batch_thgpu['sufficientSelfnonsurfacePCmaskfloat'][
                dists < datasetMetaConf['selfnonsurfaceDeltaRange']] = -2
            batch_thgpu['sufficientSelfnonsurfacePCsdfabs'] = dists

        # xyzCam and xyzCamPersp
        for samplingMethod in samplingMethodList:
            if samplingMethod in ['fullsurface']:
                continue
            camR = batch_thgpu['cam'][:, :3, :3]
            camT = batch_thgpu['cam'][:, :3, 3]
            xyzWorld = batch_thgpu['sufficient%sPCxyzWorld' % bt(samplingMethod)]
            xyzCam = torch.matmul(xyzWorld, camR.permute(0, 2, 1)) + camT[:, None, :]
            xyzCamPersp = torch.stack([
                datasetMetaConf['fScaleWidth'] * torch.div(
                    xyzCam[:, :, 0], torch.clamp(xyzCam[:, :, 2], min=datasetMetaConf['zNear'])),
                datasetMetaConf['fScaleHeight'] * torch.div(
                    xyzCam[:, :, 1], torch.clamp(xyzCam[:, :, 2], min=datasetMetaConf['zNear'])),
                xyzCam[:, :, 2],
            ], 2)
            maskfloat = batch_thgpu['sufficient%sPCmaskfloat' % bt(samplingMethod)]
            maskfloat[
                ((xyzCamPersp[:, :, 2] <= 0) | (xyzCamPersp[:, :, 2] >= 2 * r) |
                 (xyzCamPersp[:, :, 0] <= -1) | (xyzCamPersp[:, :, 0] >= 1) |
                 (xyzCamPersp[:, :, 1] <= -1) | (xyzCamPersp[:, :, 1] >= 1) |
                 (xyzCam[:, :, 0] <= -r) | (xyzCam[:, :, 0] >= r) |
                 (xyzCam[:, :, 1] <= -r) | (xyzCam[:, :, 1] >= r)
                 ) &
                (maskfloat == 1)
                ] = -1.
            batch_thgpu['sufficient%sPCxyzCam' % bt(samplingMethod)] = xyzCam
            batch_thgpu['sufficient%sPCxyzCamPersp' % bt(samplingMethod)] = xyzCamPersp
            batch_thgpu['sufficient%sPCmaskfloat' % bt(samplingMethod)] = maskfloat

            if ('sufficient%sPCgradSdfWorld' % bt(samplingMethod)) in batch_thgpu.keys():
                batch_thgpu['sufficient%sPCgradSdfCam' % bt(samplingMethod)] = torch.matmul(
                    batch_thgpu['sufficient%sPCgradSdfWorld' % bt(samplingMethod)],
                    camR.permute(0, 2, 1))

            if ('sufficient%sPCgradSdfWorld' % bt(samplingMethod)) in batch_thgpu.keys():
                batch_thgpu['sufficient%sPCgradSdfCam' % bt(samplingMethod)] = torch.matmul(
                    batch_thgpu['sufficient%sPCgradSdfWorld' % bt(samplingMethod)],
                    camR.permute(0, 2, 1))

            # from sufficient to final
            #   flagging
            tmp = batch_thgpu['sufficient%sPCmaskfloat' % bt(samplingMethod)] > 0
            #   checking
            if tmp.sum(1).min() < datasetMetaConf['numSamplingFinal%s' % bt(samplingMethod)]:
                print('    !!!!!!!!! [Fatal Error] samplingSufficient is not sufficient. Please check!!!!!!!!!')
                print('Info: samplingMethod: %s, tmp.sum(1).min(): %d, required: %d' %
                      (samplingMethod, tmp.sum(1).min(),
                       datasetMetaConf['numSamplingFinal%s' % bt(samplingMethod)]))
                # import ipdb
                # ipdb.set_trace()
                # raise ValueError('You cannot proceed.')
            #   bubble up
            tmp = torch.argsort(tmp.int(), dim=1, descending=True)[
                  :, :datasetMetaConf['numSamplingFinal%s' % bt(samplingMethod)]]
            tmp3 = tmp[:, :, None].repeat(1, 1, 3)
            #   gather all
            for k in ['xyzWorld', 'xyzCam', 'xyzCamPersp', 'gradSdfCam']:
                if 'sufficient%sPC%s' % (bt(samplingMethod), k) in batch_thgpu.keys():
                    batch_thgpu['final%sPC%s' % (bt(samplingMethod), k)] = \
                        torch.gather(
                            batch_thgpu['sufficient%sPC%s' % (bt(samplingMethod), k)], dim=1, index=tmp3)
            for k in ['occfloat', 'sdf', 'semantic']:
                if 'sufficient%sPC%s' % (bt(samplingMethod), k) in batch_thgpu.keys():
                    batch_thgpu['final%sPC%s' % (bt(samplingMethod), k)] = \
                        torch.gather(
                            batch_thgpu['sufficient%sPC%s' % (bt(samplingMethod), k)], dim=1, index=tmp)
            #   checking
            maskfloatFinal = torch.gather(
                batch_thgpu['sufficient%sPCmaskfloat' % bt(samplingMethod)], dim=1, index=tmp)
            # if not (maskfloatFinal > 0).all():
            #     print('    [Fatal Bug] maskfloat is not all zero. Please Check!')
            #     import ipdb
            #     ipdb.set_trace()
            #     raise ValueError('You cannot proceed.')
            batch_thgpu['final%sPCmaskfloat' % bt(samplingMethod)] = maskfloatFinal

            # deleting
            del camR, camT, xyzWorld, xyzCam, xyzCamPersp, tmp, tmp3, maskfloatFinal

        return batch_thgpu

    def batchPreprocessingTHGPU(self, batch_thcpu, batch_thgpu, **kwargs):
        datasets = kwargs['datasets']
        datasetMetaConf = kwargs['datasetMetaConf']
        iterCount = kwargs['iterCount']

        batch_thgpu = self.samplingBatchTHGPU(batch_thcpu,
                                              batch_thgpu,
                                              datasets=datasets,
                                              cudaDevice=self.cudaDeviceForAll,
                                              verbose=self.samplingVerbose,
                                              datasetMetaConf=datasetMetaConf,
                                              R=self.config.R,
                                              r=self.config.r,
                                              meta=self.meta,
                                              projRoot=self.projRoot,
                                              iterCount=iterCount)
        return batch_thgpu

    @classmethod
    def forwardNet(cls, batch_thgpu, **kwargs):
        datasetMetaConf = kwargs['datasetMetaConf']
        models = kwargs['models']
        c_dim_specs = kwargs['c_dim_specs']

        encoder = models['encoder']
        decoder = models['decoder']
        positionalEncoder = models['positionalEncoder']
        dgsLayer = models['dgsLayer']

        # network 1
        ss = encoder(batch_thgpu['imgForUse'].contiguous())

        for samplingMethod in ['nearsurface', 'selfnonsurface']:
            pCam = batch_thgpu['final%sPCxyzCam' % bt(samplingMethod)]
            pCamPersp = torch.stack([
                datasetMetaConf['fScaleWidth'] * torch.div(
                    pCam[:, :, 0], torch.clamp(pCam[:, :, 2], min=datasetMetaConf['zNear'])),
                datasetMetaConf['fScaleHeight'] * torch.div(
                    pCam[:, :, 1], torch.clamp(pCam[:, :, 2], min=datasetMetaConf['zNear'])),
                pCam[:, :, 2],
            ], 2)

            if samplingMethod in ['nearsurface']:
                tt = {}
                for j in [32, 16, 8, 4, 2]:
                    tt['t%d' % j] = grid_sample(
                        ss['s%d' % j], pCamPersp[:, :, None, :2].contiguous(),
                        mode='bilinear', padding_mode='border', align_corners=False
                    )[:, :, :, 0].permute(0, 2, 1)
                phi = torch.cat([tt['t' + k[1:]] for k in c_dim_specs.keys()
                                 if k.startswith('s')], 2).contiguous()
                phi = phi.view(pCamPersp.shape[0] * pCamPersp.shape[1], -1).contiguous()
                pe = positionalEncoder(pCam.view(-1, 3).contiguous(),
                                       forwardMode='valOnly').contiguous()
                lin = decoder(pe, phi, forwardMode='valOnly')
                out = torch.sigmoid(lin)
                batch_thgpu['final%sPCpredOccfloat' % bt(samplingMethod)] = \
                    out.view(pCam.shape[0], pCam.shape[1])
                del tt, phi, pe, lin, out
            elif samplingMethod in ['selfnonsurface']:
                tt4 = {}
                for j in [32, 16, 8, 4, 2]:
                    tt4['t%d' % j] = dgsLayer(ss['s%d' % j], pCamPersp).\
                        permute(0, 3, 1, 2).contiguous()
                phi4 = torch.cat([tt4['t' + k[1:]] for k in c_dim_specs.keys()
                                  if k.startswith('s')], 2).contiguous()
                phi4 = phi4.view(pCamPersp.shape[0] * pCamPersp.shape[1], -1, 4)
                pe4 = positionalEncoder(
                    pCam.view(-1, 3).contiguous(),
                    forwardMode='gradTrack',
                ).contiguous()
                lin4 = decoder(pe4, phi4, forwardMode='gradTrack')
                out0 = torch.sigmoid(lin4[..., :1])
                out3 = out0 * (1 - out0) * lin4[..., 1:]
                out4 = torch.cat([out0, out3], -1)
                batch_thgpu['final%sPCpredOccfloat' % bt(samplingMethod)] = \
                    out4[..., 0].view(pCam.shape[0], pCam.shape[1]).contiguous()
                batch_thgpu['final%sPCpredGradOccfloat' % bt(samplingMethod)] = \
                    out4[..., 1:].view(pCam.shape[0], pCam.shape[1], 3).contiguous()
            else:
                raise NotImplementedError('Unknown samplingMethod: %s' % samplingMethod)

        batch_thgpu['depthPred'] = ss['x']

        return batch_thgpu

    @classmethod
    def forwardLoss(cls, batch_thgpu, **kwargs):
        wl = kwargs['wl']
        ifRequiresGrad = kwargs['ifRequiresGrad']

        # lossDepthRegL1
        if 'lossDepthRegL1' in wl.keys() and wl['lossDepthRegL1'] > 0:
            depthRaw = batch_thgpu['depthForUse']
            depthPred = batch_thgpu['depthPred']
            mask = torch.isfinite(depthRaw)
            batch_thgpu['lossDepthRegL1'] = (depthRaw[mask] - depthPred[mask]).abs().mean()
            del depthRaw, depthPred, mask

        # lossNearsurfaceClf
        if wl['lossNearsurfaceClf'] > 0:
            tmp = F.binary_cross_entropy(
                batch_thgpu['finalNearsurfacePCpredOccfloat'],
                batch_thgpu['finalNearsurfacePCoccfloat'],
                reduction='none',
            )
            batch_thgpu['lossNearsurfaceClf'] = (
                    tmp * (batch_thgpu['finalNearsurfacePCmaskfloat'] > 0).float()
            ).mean()
            del tmp

        # lossSelfnonsurfaceGradOccfloat
        if ifRequiresGrad and 'lossSelfnonsurfaceGradOccfloat' in wl.keys() \
                and wl['lossSelfnonsurfaceGradOccfloat'] > 0:
            batch_thgpu['lossSelfnonsurfaceGradOccfloat'] = torch.abs(
                batch_thgpu['finalSelfnonsurfacePCpredGradOccfloat'] *
                (batch_thgpu['finalSelfnonsurfacePCmaskfloat'] > 0).float()[:, :, None]
            ).mean()

        # lossNearsurfaceSemantic
        if 'lossNearsurfaceSemantic' in wl.keys() and \
            wl['lossNearsurfaceSemantic'] > 0:
            mask = (batch_thgpu['finalNearsurfacePCmaskfloat'] > 0) & \
                   (batch_thgpu['finalNearsurfacePCsemantic'] >= 0)
            label = batch_thgpu['finalNearsurfacePCsemantic'].detach().clone()
            label[mask == 0] = 0
            pred = batch_thgpu['finalNearsurfacePCpredSemanticPreProb']
            label = label.view(-1)
            mask = mask.view(-1)
            pred = pred.view(mask.shape[0], -1)
            batch_thgpu['lossNearsurfaceSemantic'] = \
                F.cross_entropy(pred[mask], label.long()[mask])
            del mask, label, pred

        # lossSelfnonsurfaceGradSemantic
        if ifRequiresGrad and 'lossSelfnonsurfaceGradSemantic' in wl.keys() and \
            wl['lossSelfnonsurfaceGradSemantic'] > 0:
            batch_thgpu['lossSelfnonsurfaceGradSemantic'] = torch.abs(
                batch_thgpu['finalSelfnonsurfacePCpredGradSemanticProb'] *
                (batch_thgpu['finalSelfnonsurfacePCmaskfloat'] > 0).float()[:, :, None, None]
            ).mean()

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

        batch_thgpu = self.forwardNet(
            batch_thgpu,
            models=self.models,
            perturbDelta=config.get('perturbDelta', None),
            meta=self.meta,
            uuUsedKeys=config.get('uuUsedKeys', None),
            zzUsedKeys=config.get('zzUsedKeys', None),
            L=config.datasetMetaConf.get('gridPointLength', None),
            c_dim_specs=config.get('c_dim_specs', None),
            datasetMetaConf=config.datasetMetaConf,
            ifRequiresGrad=ifRequiresGrad,
        )
        batch_thgpu = self.forwardLoss(
            batch_thgpu,
            wl=config.wl,
            iterCount=iterCount,
            ifRequiresGrad=ifRequiresGrad,
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
            ifMonitorVal = (iterCount > self.resumeIter) & (
                    (iterCount % self.monitorValFreq == monitorMode) or
                    (iterCount < self.monitorValFreq and iterCount % (self.monitorValFreq / 2) == monitorMode))
            ifMonitorVal = (ifMonitorVal or ifMonitorDump) and (iterCount >= self.monitorValFreq)

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
            batch_thgpu = self.batchPreprocessingTHGPU(batch_thcpu, batch_thgpu,
                                                       datasets=self.datasets,
                                                       datasetMetaConf=self.config.datasetMetaConf,
                                                       iterCount=iterCount)
            if ifPrint and ifTracked:
                self.logger.info('    [Timer] batchPreprocessingTHGPU Iter%d: %.3f seconds' % (iterCount, time.time() - t))

            # ------------------------------------ main course ------------------------------------ #
            t = time.time()

            batch_thgpu = self.forwardBackwardUpdate(batch_thgpu,
                                                     ifTrain=True,
                                                     iterCount=iterCount,
                                                     datasetMetaConf=config.datasetMetaConf,
                                                     ifAllowBackward=True,
                                                     ifRequiresGrad=True)
            if ifPrint and ifTracked:
                self.logger.info('    [Timer] forwardBackwardUpdate Iter%d: %.3f seconds' % (iterCount, time.time() - t))

            # ---------------------------------------- Meta --------------------------------------- #
            if ifMonitorTrain:
                if self.numMpProcess > 0:
                    dist.barrier()

                with torch.no_grad():
                    for k in self.models:
                        if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                            self.models[k].eval()
                    if self.numMpProcess > 0:
                        dist.barrier()
                    batch_vis = castAnything(batch_thgpu, 'thgpu2np')
                    ben = self.stepMonitorTrain(batch_vis, iterCount=iterCount,
                                                ifMonitorDump=ifMonitorDump)
                    for k in ben.keys():
                        batch_thgpu['statTrain' + bt(k)] = \
                            torch.from_numpy(ben[k]).to(batch_thgpu['index'].device).mean()
                    if self.numMpProcess > 0:
                        dist.barrier()
                    for k in self.models:
                        if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                            self.models[k].train()

                if self.numMpProcess > 0:
                    dist.barrier()

            # ifMonitorVal = False
            if ifMonitorVal:
                if self.numMpProcess > 0:
                    dist.barrier()

                with torch.no_grad():
                    for k in self.models:
                        if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                            self.models[k].eval()
                    if self.numMpProcess > 0:
                        dist.barrier()
                    ben = self.stepMonitorVal(iterCount=iterCount,
                                              ifMonitorDump=ifMonitorDump)
                    for k in ben.keys():
                        batch_thgpu['statVal' + bt(k)] = \
                            torch.from_numpy(ben[k]).to(batch_thgpu['index'].device).mean()
                    if self.numMpProcess > 0:
                        dist.barrier()
                    for k in self.models:
                        if k != 'meta' and k not in self.models['meta']['nonLearningModelNameList']:
                            self.models[k].train()

                if self.numMpProcess > 0:
                    dist.barrier()

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

                # visualVal forwarding and saving (not including visualization)
                # self.saveValBatchVis(iterCount=iterCount)
            else:
                del batch_thgpu

            if ifBackupTheLatestModels and ifTracked:
                self.saveModelSnapshot(iterCount='latest')

            if ifPrint and ifTracked:
                self.logger.info('[Timer] Total is %.3f seconds.' % (time.time() - timeIterStart))

            iterCount += 1
            pass

    def train(self):
        # self.trainNoException()
        try:
            self.trainNoException()
        except KeyboardInterrupt as e:
            self.logger.info('KeyboardInterrupt Detected: Rank %d' % self.rank)
            if self.rank == self.trackedRank and self.finishedIterCount > 10000:
                self.saveModelSnapshot(iterCount=self.finishedIterCount)
                self.saveBatchVis(self.finishedBatchVis, iterCount=self.finishedIterCount)
                # self.saveValBatchVis(iterCount=self.finishedIterCount)
            dist.barrier()
            dist.destroy_process_group()


