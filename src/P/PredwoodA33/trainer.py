# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (tfconda)

# PredwoodA32/33
# The same functionality with PredwoodA30, just that rename the configurations. (even numbers for hm, odd numbers for scannet)
# Here we use pydgs instead of dgs due to unstable training speed of our own cuda dgs.

# PredwoodA30
#   - Compared to PredwoodA22, Now we use hm3d to replace scannet to see how the performance would look like.

# PredwoodA22
#   - Compared to PredwoodA2, Now we use the "Bpredwood2" benchmarking routine.

# Basic
import functools
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
import pickle
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
from skimage.io import imsave

# toolbox
from UDLv3 import udl
from codes_py.toolbox_framework.framework_util_v4 import \
    checkPDSRLogDirNew, castAnything, probe_load_network, load_network, save_network, \
    bsv02bsv, mergeFromAnotherBsv0, bsv2bsv0, constructInitialBatchStepVis0, \
    batchExtractParticularDataset
from codes_py.toolbox_3D.depth_v1 import depthMap2mesh
from codes_py.toolbox_torch.hook_v1 import PyTorchForwardHook
from codes_py.np_ext.mat_io_v1 import pSaveMat
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoTHGPU, vertInfo2faceVertInfoNP
from codes_py.toolbox_3D.sdf_from_multiperson_v1 import sdfTHGPU
from pytorch3d.ops.knn import knn_points
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
from codes_py.toolbox_3D.rotations_v1 import camSys2CamPerspSysTHGPU
from codes_py.toolbox_3D.rotations_v1 import camSys2CamPerspSys0
from codes_py.toolbox_3D.mesh_surgery_v1 import addPointCloudToMesh
from codes_py.toolbox_show_draw.html_v1 import HTMLStepperNoPrinting
from codes_py.np_ext.data_processing_utils_v1 import determinedArbitraryPermutedVector2correct
from codes_py.toolbox_3D.self_sampling_v1 import mesh_sampling_given_normal_np_simple
from codes_py.toolbox_3D.pyrender_wrapper_v1 import PyrenderManager
# from codes_py.toolbox_3D.dgs_wrapper_v1 import DGS2DLayer
from codes_py.toolbox_3D.pydgs_v1 import pydgs_forwardValOnly
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_3D.view_query_generation_v1 import gen_viewport_query_given_bound
from codes_py.toolbox_3D.representation_v1 import voxSdfSign2mesh_skmc, directDepthMap2PCNP
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly, dumpPly2, dumpPlyPointCloud
from codes_py.toolbox_3D.mesh_surgery_v1 import trimVert, addPointCloudToMesh, \
    combineMultiShapes_withVertRgb, create_lineArrow_mesh

# PDSR
from .dataset import PScannetGivenRenderDataset, POmnidataBerkeleyDataset
# from .lib.multi_depth_model_woauxi import DepthModel
from .midas.dpt_depth import DPTDepthModel
from .resnetfc import ResNetFC
from .positionalEncoding import PositionalEncoding
from .losses import virtual_normal_loss, midas_loss

# benchmarking
from Bpredwood2.testDataEntry.scannetGivenRenderDataset import ScannetMeshCache, collate_fn_scannetGivenRender
from Bpredwood2.testDataEntry.testDataEntryPool import getTestDataEntryDict
from Bpredwood2.csvGeometry3DEntry.benchmarkingGeometry3DScannet \
    import benchmarkingGeometry3DScannetFunc
# from Bpredwood2.csvGeometry3DEntry.benchmarkingGeometry3DHm import benchmarkingGeometry3DHmFunc
from Bpredwood2.csvGeometry3DEntry.dumpHtmlForPrepickGeometry3D import addToSummary0Txt0BrInds

# inlines
bt = lambda s: s[0].upper() + s[1:]


def printInfo(rank, tag):
    # print('[%d] Reaching %s.' % (rank, tag))
    pass


class Trainer(object):
    @staticmethod
    def setup_for_distributed(is_master):
        """
        This function disables printing when not in master process
        """
        pass

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

        ''' Unless you are debugging, fixing random seed for learning from such a big dataset seems not very helpful (ctrl-C would restart the dataloader iteration)
        # enable them only for debugging purpose (fix random seed to chase a particular data point)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        '''

        # cuda backend
        torch.backends.cudnn.benchmark = True

        # meta params (not influencing model training)
        self.printFreq = 10
        self.minSnapshotFreq = 2000
        self.samplingVerbose = False
        self.monitorTrainFreq = 2000
        self.monitorValFreq = 20000
        self.monitorDumpFreq = 20000
        self.monitorMode = 80  # which iteration is the first to step the monitors

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
            if datasetConf['class'] == 'ScannetGivenRenderDataset':
                datasetObj = PScannetGivenRenderDataset(
                    datasetConf,
                    datasetSplit=datasetConf['trainingSplit'],
                    projRoot=projRoot,
                )
                batchSizePerProcess = datasetConf['batchSizePerProcess']
                num_workers = int(math.ceil(batchSizePerProcess / 4.))
                dataLoader = torch.utils.data.DataLoader(
                    datasetObj, batch_size=batchSizePerProcess, num_workers=num_workers,
                    shuffle=True, drop_last=True, collate_fn=collate_fn_scannetGivenRender,
                )
                datasetObjDict[datasetConf['dataset']] = datasetObj
                dataLoaderDict[datasetConf['dataset']] = dataLoader
                dataLoaderIterDict[datasetConf['dataset']] = iter(dataLoader)
            elif datasetConf['class'] == 'OmnidataBerkeleyDataset':
                trainsets = OrderedDict(
                    (componentName, POmnidataBerkeleyDataset(
                        datasetConf, projRoot=self.projRoot,
                        datasetSplit=datasetConf['trainingSplit'],
                        componentName=componentName,
                        ifConductDataAugmentation=True,
                    )) for componentName in datasetConf['componentFrequency'].keys())

                trainsets_lens = [len(trainsets[k])
                                  for k in datasetConf['componentFrequency'].keys()]
                trainsets_weights = [
                    datasetConf['componentFrequency'][k] / len(trainsets[k])
                    for k in datasetConf['componentFrequency'].keys()
                ]
                sampler_weights = []
                for w, count in zip(trainsets_weights, trainsets_lens):
                    sampler_weights += [w] * count
                sampler_weights = torch.tensor(sampler_weights)
                sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))
                concatenatedTrainsets = ConcatDataset(trainsets.values())
                batchSizePerProcess = datasetConf['batchSizePerProcess']
                num_workers = int(math.ceil(batchSizePerProcess / 4.))
                dataLoader = DataLoader(
                    concatenatedTrainsets, batch_size=batchSizePerProcess, num_workers=num_workers,
                    drop_last=True, sampler=sampler, pin_memory=False
                )
                for componentName in trainsets.keys():
                    datasetObjDict[datasetConf['dataset'] + '_' + componentName] = \
                        trainsets[componentName]
                dataLoaderDict[datasetConf['dataset']] = dataLoader
                dataLoaderIterDict[datasetConf['dataset']] = iter(dataLoader)
                del trainsets, trainsets_lens, trainsets_weights, \
                    sampler_weights, sampler, batchSizePerProcess, num_workers, \
                    concatenatedTrainsets
            else:
                raise NotImplementedError('Unknown dataset class: %s' % datasetConf['class'])
        self.datasetObjDict = datasetObjDict
        self.dataLoaderDict = dataLoaderDict
        self.dataLoaderIterDict = dataLoaderIterDict

    def _netConstruct(self, **kwargs):
        config = self.config
        self.logger.info('[Trainer] MetaModelLoading - _netConstruct')
        projRoot = self.projRoot

        # encoder = DepthModel(encoder=config.depthEncoderTag)
        encoder = DPTDepthModel(
            path=None,  # Use our own code to do finetuning loading
            non_negative=True,
            features=256,
            backbone=config.omnitoolsBackbone,
            readout='project',
            channels_last=False,
            use_bn=False,
        )

        if config.ifFinetunedFromOmnitools == 0:
            pass  # using the default initialization
        elif config.ifFinetunedFromOmnitools == 1:
            if config.omnitoolsBackbone in ['vitb_rn50_384']:
                omnitools_pretrained_weight_file = projRoot + \
                                                   'external_codes/omnidata_tools/omnidata_tools/' \
                                                   'torch/pretrained_models/omnidata_rgb2depth_dpt_hybrid.pth'
            elif config.omnitoolsBackbone in ['vitl16_384']:
                omnitools_pretrained_weight_file = projRoot + \
                                                   'external_codes/omnidata_tools/omnidata_tools/' \
                                                   'torch/pretrained_models/omnidata_rgb2depth_dpt_large.pth'
            else:
                raise NotImplementedError('Unknown config.adelaiHalfEncoderTag: %s' %
                                          config.adelaiHalfEncoderTag)
            sd = torch.load(
                omnitools_pretrained_weight_file, map_location='cpu')
            encoder.load_state_dict(sd, strict=False)
        else:
            raise NotImplementedError('Unknown ifFinetunedFromAdelai: %d' % config.ifFinetunedFromAdelai)

        c_dim = sum(list(config.c_dim_specs.values()))
        decoder = ResNetFC(d_in=39, d_out=1, n_blocks=5, d_latent=c_dim, d_hidden=512, beta=0.0,
                           combine_layer=3)
        positionalEncoder = PositionalEncoding(
            num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True)
        # dgs2dLayer = DGS2DLayer()
        meta = {
            'nonLearningModelNameList': ['positionalEncoder', 'dgs2dLayer'],
            'c_dim_specs': config.c_dim_specs,
        }
        self.models = dict()
        self.models['decoder'] = decoder.to(self.cudaDeviceForAll)
        self.models['encoder'] = encoder.to(self.cudaDeviceForAll)
        self.models['positionalEncoder'] = positionalEncoder.to(self.cudaDeviceForAll)
        # self.models['dgs2dLayer'] = dgs2dLayer.to(self.cudaDeviceForAll)
        self.models['meta'] = meta

    def _netRandomInitialization(self, iter_label):
        # random initialization
        # (Different threads might result in different model weights. Only Rank 0 is used)
        self.logger.info('[Trainer] MetaModelLoading - _netRandomInitialization')

        '''
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
        '''
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
                        self.logDir, self.models[k], k, resumeIter, map_location=self.cudaDeviceForAll)
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
        datasetConf_omnidataBerkeley = self.config.datasetConfDict['omnidataBerkeley']
        self.vnl_loss = virtual_normal_loss.VNL_Loss(
            1.0, 1.0, (datasetConf_omnidataBerkeley['winHeight'], datasetConf_omnidataBerkeley['winWidth']))
        self.midas_loss = midas_loss.MidasLoss(alpha=0.1)

    def setupMetaVariables(self):
        meta = {}

        meta = self._meta_initialize_scannet_house_vert_world_cache(meta)
        self.meta = meta

    @staticmethod
    def _meta_initialize_scannet_house_vert_world_cache(meta):
        meta['scannetMeshCache'] = ScannetMeshCache()
        return meta

    def setupMonitor(self):
        self.monitorImmediateFlowLogDir = self.logDir + 'monitorImmediateFlow/'
        self.monitorTrainLogDir = self.logDir + 'monitorTrain/'
        self.monitorValLogDir = self.logDir + 'monitorVal/'
        self.htmlStepperImmediateFlow = HTMLStepper(self.monitorImmediateFlowLogDir, 100, 'monitorImmediateFlow')
        self.htmlStepperTrain = HTMLStepper(self.monitorTrainLogDir, 100, 'monitorTrain')
        self.htmlStepperVal = HTMLStepper(self.monitorValLogDir, 100, 'monitorVal')
        self.testDataEntry = getTestDataEntryDict(
            wishedTestDataNickName=['scannetOfficialTestSplit10'])['scannetOfficialTestSplit10']

    def initializeAll(self, **kwargs):
        # kwargs
        iter_label = kwargs['iter_label']  # required for all cases
        hook_type = kwargs['hook_type']  # required for all cases

        self.setupMetaVariables()
        self.metaDataLoading()
        self.resumeIter = self.metaModelLoading(iter_label=iter_label,
                                                hook_type=hook_type)
        self.metaLossPrepare()
        self.setupMonitor()

        self.logger.info('Initialization finished! Rank = %d' % self.rank)

    def saveAnomaly(self, **kwargs):
        message = kwargs['message']
        iterCount = kwargs['iterCount']
        batch_thgpu = kwargs['batch_thgpu']
        with open(self.logDir + 'anomaly/%s_thgpu_host_%s_iter_%d_rank_%d.pkl' %
                (message, gethostname(), iterCount, self.rank), 'wb') as f:
            pickle.dump(batch_thgpu, f)
        models = self.models
        for k in models.keys():
            if (k != 'meta') and (k not in models['meta']['nonLearningModelNameList']):
                if self.numMpProcess <= 0:
                    torch.save(
                        self.models[k].state_dict(),
                        self.logDir + 'anomaly/%s_models%s_host_%s_iter_%d_rank_%d.pth' %
                            (message, bt(k), gethostname(), iterCount, self.rank)
                    )
                else:
                    torch.save(
                        self.models[k].module.state_dict(),
                        self.logDir + 'anomaly/%s_models%s_host_%s_iter_%d_rank_%d.pth' %
                            (message, bt(k), gethostname(), iterCount, self.rank)
                    )

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

    def stepMonitorImmediateFlow(self, batch_vis, **kwargs):  # use the current training data for immediate use (e.g. with augmentation)
        iterCount = kwargs['iterCount']
        ifMonitorDump = kwargs['ifMonitorDump']
        ifIsTrackedRank = self.rank == self.trackedRank
        pyrenderManager = kwargs['pyrenderManager']
        htmlStepper = self.htmlStepperImmediateFlow

        config = self.config
        P = config['P']
        D = config['D']
        S = config['S']
        R = config['R']

        testDataEntry = self.testDataEntry
        datasetObj = testDataEntry['datasetObj']

        ben = {}
        batch_vis_now = batchExtractParticularDataset(batch_vis, 'scannetGivenRender')
        for visIndex in range(min(4, batch_vis_now['index'].shape[0])):
            print('    Stepping Visualizer ImmediateFlow: %d' % visIndex)
            assert testDataEntry['datasetObj'].dataset == 'scannetGivenRender'
            bsv0 = bsv2bsv0(batch_vis_now, visIndex)  # contains all augmentation
            # bsv0 = testDataEntry['datasetObj'].getOneNP(
            #     int(batch_vis_now['index'][visIndex]))
            bsv0_initial = constructInitialBatchStepVis0(
                bsv02bsv(bsv0), iterCount=iterCount, visIndex=0, dataset=None,
                P=P, D=D, S=S, R=R,
                verboseGeneral=0,
            )
            bsv0_initial = mergeFromAnotherBsv0(
                bsv0_initial, bsv0,
                copiedKeys=list(set(bsv0.keys()) - set(bsv0_initial.keys()))
            )

            ifRequiresDrawing = (visIndex < 4) and ifIsTrackedRank
            bsv0 = benchmarkingGeometry3DScannetFunc(
                bsv0_initial,
                scannetMeshCache=self.meta['scannetMeshCache'],
                datasetObj=datasetObj,
                # misc
                cudaDevice=self.cudaDeviceForAll,
                raiseOrTrace='ignoreAndNone',
                # benchmarking rules
                voxSize=128, numMeshSamplingPoint=200000,
                # drawing
                ifRequiresDrawing=ifIsTrackedRank,
                pyrenderManager=pyrenderManager,
                # predicting
                ifRequiresPredictingHere=True,
                models=self.models, verboseBatchForwarding=0,
                Trainer=type(self), doPred0Func=self.doPred0,
                useSurfaceOrDepth='surface', meta=self.meta,
            )
            if bsv0 is None:
                continue
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
                subMessage = 'FitF1: %.3f' % bsv0['finalBenFitF1']
                htmlStepper.step2(summary0, txt0, brInds, headerMessage, subMessage)
            if ifIsTrackedRank and ifMonitorDump and ifRequiresDrawing:
                # dump the mesh
                for meshName in ['evalPredFit', 'evalViewFit', 'view']:
                    sysLabel = 'cam'
                    dumpPly(
                        self.monitorImmediateFlowLogDir + '%s%s%s%sI%d_%s%s_%s_%d(%d).ply' %
                            (P, D, S, R, iterCount, meshName, bt(sysLabel),
                             bsv0['dataset'], bsv0['index'], bsv0['flagSplit']),
                        bsv0['%sVert%s' % (meshName, bt(sysLabel))],
                        bsv0['%sFace' % meshName])

                # overlay (with normal) (do not dump the mesh - use meshlab joint mesh visualization)
                sysLabel = 'cam'
                pcInfoTupList = [  # (pcName, pointColor, arrowColor)
                    ('finalUnarysurface', [1, 1, 0], [1, 0, 0]),
                    ('finalLocalEdgePairwiseSurfaceA', [1, 1, 0], [0.5, 0, 0.2]),
                    ('finalLocalEdgePairwiseSurfaceB', [0, 1, 0], [0, 0.5, 0.2]),
                    ('finalLocalPlanarPairwiseSurfaceA', [1, 1, 0], [0.5, 0, 0.2]),
                    ('finalLocalPlanarPairwiseSurfaceB', [0, 1, 0], [0, 0.5, 0.2]),
                ]
                arrow_rLine = 0.003
                arrow_rArrow = 0.01
                arrow_length = 0.3
                point_r = 0.02
                for pcInfoTup in pcInfoTupList:
                    pcName, pointColor, arrowColor = pcInfoTup
                    if '%sPCxyz%s' % (pcName, bt(sysLabel)) in bsv0.keys():
                        pcXyz0 = bsv0['%sPCxyz%s' % (pcName, bt(sysLabel))]
                        pcNormal0 = bsv0['%sPCnormal%s' % (pcName, bt(sysLabel))]
                        vfc = combineMultiShapes_withVertRgb([
                            create_lineArrow_mesh(
                                arrow_rLine, arrow_rArrow, pcXyz0[q, :],
                                pcXyz0[q, :] + arrow_length * pcNormal0[q, :],
                                arrowColor,
                            )
                            for q in range(500)  # pcXyz0.shape[0])
                        ])
                        vfc = addPointCloudToMesh(
                            vfc, pcXyz0, pointColor, 'sphere', point_r, False
                        )
                        dumpPly(
                            self.monitorImmediateFlowLogDir + '%s%s%s%sI%d_%s%s_%s_%d(%d).ply' %
                                (P, D, S, R, iterCount, pcName, bt(sysLabel),
                                 bsv0['dataset'], bsv0['index'], bsv0['flagSplit']),
                            vfc[0], vfc[1], vfc[2],
                        )

                # If you want to dump other styled point cloud (e.g. nearsurface w/ red/blue w/o/ grad)
                # Do it here.

        tmp = {'finalBenValidCount': np.array(len(ben['finalBenFitF1']), dtype=np.float32)}
        tmp.update(
            {k: np.array(ben[k], dtype=np.float32) for k in ben.keys()})
        ben = tmp
        return ben

    def stepMonitorTrain(self, batch_vis, **kwargs):  # treat the training sample as a test case (no augmentation)
        iterCount = kwargs['iterCount']
        ifMonitorDump = kwargs['ifMonitorDump']
        ifIsTrackedRank = self.rank == self.trackedRank
        pyrenderManager = kwargs['pyrenderManager']
        htmlStepper = self.htmlStepperTrain

        config = self.config
        P = config['P']
        D = config['D']
        S = config['S']
        R = config['R']

        testDataEntry = self.testDataEntry
        datasetObj = testDataEntry['datasetObj']

        ben = {}
        batch_vis_now = batchExtractParticularDataset(batch_vis, 'scannetGivenRender')
        for visIndex in range(min(4, batch_vis_now['index'].shape[0])):
            print('    Stepping Visualizer Train: %d' % visIndex)
            assert testDataEntry['datasetObj'].dataset == 'scannetGivenRender'
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
            bsv0 = benchmarkingGeometry3DScannetFunc(
                bsv0_initial,
                scannetMeshCache=self.meta['scannetMeshCache'],
                datasetObj=datasetObj,
                # misc
                cudaDevice=self.cudaDeviceForAll,
                raiseOrTrace='ignoreAndNone',
                # benchmarking rules
                voxSize=128, numMeshSamplingPoint=200000,
                # drawing
                ifRequiresDrawing=ifIsTrackedRank,
                pyrenderManager=pyrenderManager,
                # predicting
                ifRequiresPredictingHere=True,
                models=self.models, verboseBatchForwarding=0,
                Trainer=type(self), doPred0Func=self.doPred0,
                useSurfaceOrDepth='surface', meta=self.meta,
            )
            if bsv0 is None:
                continue
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
                subMessage = 'FitF1: %.3f' % bsv0['finalBenFitF1']
                htmlStepper.step2(summary0, txt0, brInds, headerMessage, subMessage)
            if ifIsTrackedRank and ifMonitorDump and ifRequiresDrawing:
                for meshName in ['evalPredFit', 'evalViewFit']:
                    sysLabel = 'cam'
                    dumpPly2(self.monitorTrainLogDir +
                             '%s%s%s%sI%d_%s%s_%s_%d(%d).ply' %
                             (P, D, S, R, iterCount, meshName, bt(sysLabel),
                              bsv0['dataset'], bsv0['index'], bsv0['flagSplit']),
                             bsv0['%sVert%s' % (meshName, bt(sysLabel))],
                             bsv0['%sFace' % meshName],
                             bsv0['%sFaceRgb' % meshName])
        tmp = {'finalBenValidCount': np.array(len(ben['finalBenFitF1']), dtype=np.float32)}
        tmp.update(
            {k: np.array(ben[k], dtype=np.float32) for k in ben.keys()})
        ben = tmp
        return ben

    def stepMonitorVal(self, **kwargs):
        iterCount = kwargs['iterCount']
        ifMonitorDump = kwargs['ifMonitorDump']
        ifIsTrackedRank = self.rank == self.trackedRank
        pyrenderManager = kwargs['pyrenderManager']
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
            if testDataNickName in ['scannetOfficialTestSplit10']:
                bsv0 = benchmarkingGeometry3DScannetFunc(
                    bsv0_initial,
                    scannetMeshCache=self.meta['scannetMeshCache'],
                    datasetObj=datasetObj,
                    # misc
                    cudaDevice=self.cudaDeviceForAll,
                    raiseOrTrace='ignoreAndNone',
                    # benchmarking rules
                    voxSize=128, numMeshSamplingPoint=200000,
                    # drawing
                    ifRequiresDrawing=(index in indVisChosen) and (rank == trackedRank),
                    pyrenderManager=pyrenderManager,
                    # predicting
                    ifRequiresPredictingHere=True,
                    models=self.models, verboseBatchForwarding=0,
                    Trainer=type(self), doPred0Func=self.doPred0,
                    useSurfaceOrDepth='surface', meta=self.meta,
                )
            elif testDataNickName in ['demo1', 'freedemo1']:
                bsv0 = benchmarkingDemoSingleViewReconstruction(
                    copy.deepcopy(bsv0_initial),
                    datasetObj=datasetObj,
                    # misc
                    cudaDevice=self.cudaDeviceForAll,
                    raiseOrTrace='ignoreAndNone',
                    # benchmarking rules
                    voxSize=256, numMeshSamplingPoint=200000,
                    # drawing
                    ifRequiresDrawing=(index in indVisChosen) and (rank == trackedRank),
                    pyrenderManager=pyrenderManager,
                    # predicting
                    ifRequiresPredictingHere=True,
                    models=self.models, verboseBatchForwarding=0,
                    Trainer=type(self), doPred0Func=self.doPred0,
                    useSurfaceOrDepth='surface', meta=self.meta,
                )
            else:
                raise NotImplementedError('Unknown testDataNickName: %s' % testDataNickName)
            if bsv0 is None:
                continue
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
                subMessage = 'FitF1: %.3f' % bsv0['finalBenFitF1']
                htmlStepper.step2(summary0, txt0, brInds, headerMessage, subMessage)
            if ifMonitorDump and ifIsTrackedRank and ifRequiresDrawing:
                for meshName in ['evalPredFit', 'evalViewFit']:
                    sysLabel = 'cam'
                    dumpPly2(self.monitorValLogDir +
                             '%s%s%s%sI%d_%s%s_%s_%d(%d).ply' %
                             (P, D, S, R, iterCount, meshName, bt(sysLabel),
                              bsv0['dataset'], bsv0['index'], bsv0['flagSplit']),
                             bsv0['%sVert%s' % (meshName, bt(sysLabel))],
                             bsv0['%sFace' % meshName],
                             bsv0['%sFaceRgb' % meshName])

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
        iterCount = kwargs['iterCount']
        config = self.config
        tmp = OrderedDict([])
        for did, (dataset, datasetConf) in enumerate(config.datasetConfDict.items()):
            printInfo(self.rank, 'Before loading from %s' % dataset)
            try:
                batchDid_thcpu = next(self.dataLoaderIterDict[dataset])
            except:
                self.dataLoaderIterDict[dataset] = iter(self.dataLoaderDict[dataset])
                batchDid_thcpu = next(self.dataLoaderIterDict[dataset])
            tmp[dataset] = batchDid_thcpu

        printInfo(self.rank, 'Before putting data into gpu')
        batch_thcpu = {}
        for dataset in tmp.keys():
            for k in tmp[dataset].keys():
                batch_thcpu[k + '_' + dataset] = tmp[dataset][k]

        batch_thgpu = castAnything(batch_thcpu, 'thcpu2thgpu', device=self.cudaDeviceForAll)
        return batch_thcpu, batch_thgpu

    @classmethod
    def samplingBatchForScannetTHGPU(cls, batch_thcpu, batch_thgpu, **kwargs):
        datasetObj=kwargs['datasetObj']
        meta = kwargs['meta']
        cudaDevice = kwargs['cudaDevice']
        iterCount = kwargs['iterCount']
        logDir = kwargs.get('logDir', None)

        datasetConf = datasetObj.datasetConf
        dataset = datasetObj.dataset
        _dataset = '_' + dataset
        samplingMethodList = datasetConf['samplingMethodList']
        B = int(batch_thcpu['index' + _dataset].shape[0])
        # minBoundWorld_thgpu = batch_thgpu['boundMinWorld' + _dataset]
        # maxBoundWorld_thgpu = batch_thgpu['boundMaxWorld' + _dataset]
        # minBoundCam_thgpu = batch_thgpu['boundMinCam' + _dataset]
        # maxBoundCam_thgpu = batch_thgpu['boundMaxCam' + _dataset]

        # cache load mesh and sample face Centroid
        finalFullsurfacePCxyzWorld = np.zeros((
            B, datasetConf['numSamplingFinalFullsurface'], 3), dtype=np.float32)
        finalFullsurfacePCnormalWorld = np.zeros((
            B, datasetConf['numSamplingFinalFullsurface'], 3), dtype=np.float32)
        finalFullsurfacePCpix3d10ID = np.zeros((
            B, datasetConf['numSamplingFinalFullsurface'], ), dtype=np.int32)
        minBoundWorld = np.zeros((B, 3), dtype=np.float32)
        maxBoundWorld = np.zeros((B, 3), dtype=np.float32)

        # debugMeshCollector = []
        for j in range(B):
            houseID0 = int(batch_thcpu['houseID' + _dataset][j])

            # cache loading
            tmp0 = meta['scannetMeshCache'].call_cache_scannet_house_vert_world_0(
                houseID0=houseID0,
                scannetFile=datasetObj.fileList_house[houseID0],
                scannetScanID=datasetObj.scanIDList_house[houseID0],
                verbose=(iterCount >= 100),
                original_dataset_root=datasetObj.original_dataset_root,
            )
            vertWorld0 = tmp0['vertWorld0']
            face0 = tmp0['face0']
            faceNormalWorld0 = tmp0['faceNormalWorld0']

            nFace = int(face0.shape[0])
            faceFlag0 = batch_thcpu['faceFlagUntrimmed' + _dataset][j][:nFace].numpy().astype(bool)
            face0 = face0[faceFlag0]
            faceNormalWorld0 = faceNormalWorld0[faceFlag0]
            faceVertWorld0 = vertInfo2faceVertInfoNP(vertWorld0[None], face0[None])[0]
            cudaDevice = batch_thgpu['index' + _dataset].device
            tmp1 = mesh_sampling_given_normal_np_simple(
                datasetConf['numSamplingFinalFullsurface'], faceVertWorld0)
            finalFullsurfacePCxyzWorld[j, :, :] = tmp1['point0']
            finalFullsurfacePCnormalWorld[j, :, :] = faceNormalWorld0[tmp1['pointFace0'], :]
            minBoundWorld[j, :] = tmp1['point0'].min(0)
            maxBoundWorld[j, :] = tmp1['point0'].max(0)

            # debug
            # vertWorld0, face0 = trimVert(vertWorld0, face0)
            # vertRgb0 = 0.6 * np.ones_like(vertWorld0)
            # vfc = (vertWorld0, face0, vertRgb0)
            # debugMeshCollector.append(vfc)

        minBoundWorld_thgpu = torch.from_numpy(minBoundWorld).to(cudaDevice)
        maxBoundWorld_thgpu = torch.from_numpy(maxBoundWorld).to(cudaDevice)

        del houseID0, tmp0, vertWorld0, face0, faceNormalWorld0, nFace
        del faceFlag0, tmp1

        if 'fullsurface' in samplingMethodList:
            batch_thgpu['finalFullsurfacePCxyzWorld' + _dataset] = \
                torch.from_numpy(finalFullsurfacePCxyzWorld).to(cudaDevice)
            batch_thgpu['finalFullsurfacePCnormalWorld' + _dataset] = \
                torch.from_numpy(finalFullsurfacePCnormalWorld).to(cudaDevice)
            batch_thgpu['finalFullsurfacePCsemantic' + _dataset] = \
                torch.from_numpy(finalFullsurfacePCpix3d10ID).to(cudaDevice)

        if 'unarysurface' in samplingMethodList:  # for unary normal loss
            batch_thgpu['sufficientUnarysurfacePCxyzWorld' + _dataset] = \
                batch_thgpu['finalFullsurfacePCxyzWorld' + _dataset][
                :, -datasetConf['numSamplingSufficientUnarysurface']:, :]
            batch_thgpu['sufficientUnarysurfacePCnormalWorld' + _dataset] = \
                batch_thgpu['finalFullsurfacePCnormalWorld' + _dataset][
                :, -datasetConf['numSamplingSufficientUnarysurface']:, :]
            batch_thgpu['sufficientUnarysurfacePCmaskfloat' + _dataset] = torch.ones(
                B, datasetConf['numSamplingSufficientUnarysurface'],
                dtype=torch.float32, device=cudaDevice)

        if ('localEdgePairwiseSurface' in samplingMethodList) or \
                ('localPlanarPairwiseSurface' in samplingMethodList):
            numNearestNeighbour = datasetConf['fullsurfaceNumNearestNeighbour']
            # obtain nearest neighbours for finalFullsurface
            tmp = knn_points(
                batch_thgpu['finalFullsurfacePCxyzWorld' + _dataset],  # (B, Q, 3(xyz))
                batch_thgpu['finalFullsurfacePCxyzWorld' + _dataset],
                K=numNearestNeighbour + 1, return_nn=True, return_sorted=True,
            )  # K: +1, mainly to rule out the identical point as the pair
            # knn = tmp.knn[:, :, 1:, :]  # (B, Q, numNearestNeighbour, 3(xyz))
            idx = tmp.idx[:, :, 1:]  # (B, Q, numNearestNeighbour)

            # pick only one neighbour
            pick = torch.randint(
                low=0, high=numNearestNeighbour, size=(B, datasetConf['numSamplingFinalFullsurface']),
                dtype=torch.int64, device=idx.device
            )
            idx = torch.gather(idx, dim=2, index=pick[:, :, None])[:, :, 0]  # (B, Q)
            idx3 = idx[:, :, None].repeat(1, 1, 3)  # (B, Q, 3)
            # (B, Q, 3)
            batch_thgpu['finalFullsurfacePCnnXyzWorld' + _dataset] = torch.gather(
                batch_thgpu['finalFullsurfacePCxyzWorld' + _dataset], dim=1, index=idx3)
            # (B, Q, 3)
            batch_thgpu['finalFullsurfacePCnnNormalWorld' + _dataset] = torch.gather(
                batch_thgpu['finalFullsurfacePCnormalWorld' + _dataset], dim=1, index=idx3)
            # (B, Q)
            batch_thgpu['finalFullsurfacePCnnIdx' + _dataset] = idx

            # (B, Q)
            batch_thgpu['finalFullsurfacePCnnDot' + _dataset] = \
                (batch_thgpu['finalFullsurfacePCnormalWorld' + _dataset] *
                    batch_thgpu['finalFullsurfacePCnnNormalWorld' + _dataset]).sum(2)
            # (B, Q)
            batch_thgpu['finalFullsurfacePCnnIdx' + _dataset] = idx
            del numNearestNeighbour, tmp, idx, idx3, pick

        if 'localEdgePairwiseSurface' in samplingMethodList:  # as sharp as possible (leps)
            dot = batch_thgpu['finalFullsurfacePCnnDot' + _dataset]
            nnIdx = batch_thgpu['finalFullsurfacePCnnIdx' + _dataset]
            tmp = torch.argsort(((dot >= -0.8) & (dot <= 0.8)).int(), dim=1, descending=True)\
                [:, :datasetConf['numSamplingSufficientLocalEdgePairwiseSurface']]
            idxA = tmp
            idxA3 = idxA[:, :, None].repeat(1, 1, 3)
            batch_thgpu['sufficientLocalEdgePairwiseSurfaceAPCxyzWorld' + _dataset] = torch.gather(
                batch_thgpu['finalFullsurfacePCxyzWorld' + _dataset], dim=1, index=idxA3)
            batch_thgpu['sufficientLocalEdgePairwiseSurfaceAPCnormalWorld' + _dataset] = torch.gather(
                batch_thgpu['finalFullsurfacePCnormalWorld' + _dataset], dim=1, index=idxA3)
            batch_thgpu['sufficientLocalEdgePairwiseSurfaceAPCmaskfloat' + _dataset] = torch.ones(
                (B, datasetConf['numSamplingSufficientLocalEdgePairwiseSurface']),
                dtype=torch.float32, device=dot.device)
            batch_thgpu['sufficientLocalEdgePairwiseSurfaceAPCnnDot' + _dataset] = torch.gather(
                dot, dim=1, index=idxA)  # debugging purpose
            batch_thgpu['sufficientLocalEdgePairwiseSurfaceAPCnnIdx' + _dataset] = torch.gather(
                nnIdx, dim=1, index=idxA)  # debugging purpose
            idxB = torch.gather(nnIdx, dim=1, index=tmp)
            idxB3 = idxB[:, :, None].repeat(1, 1, 3)
            batch_thgpu['sufficientLocalEdgePairwiseSurfaceBPCxyzWorld' + _dataset] = torch.gather(
                batch_thgpu['finalFullsurfacePCxyzWorld' + _dataset], dim=1, index=idxB3)
            batch_thgpu['sufficientLocalEdgePairwiseSurfaceBPCnormalWorld' + _dataset] = torch.gather(
                batch_thgpu['finalFullsurfacePCnormalWorld' + _dataset], dim=1, index=idxB3)
            batch_thgpu['sufficientLocalEdgePairwiseSurfaceBPCmaskfloat' + _dataset] = torch.ones(
                (B, datasetConf['numSamplingSufficientLocalEdgePairwiseSurface']),
                dtype=torch.float32, device=dot.device)

            del dot, nnIdx, tmp, idxA, idxA3, idxB, idxB3

        if 'localPlanarPairwiseSurface' in samplingMethodList:  # random (some might not be planar) (lpps)
            dot = batch_thgpu['finalFullsurfacePCnnDot' + _dataset]
            nnIdx = batch_thgpu['finalFullsurfacePCnnIdx' + _dataset]
            tmp = torch.argsort((dot > 0.8).int(), dim=1, descending=True)\
                [:, :datasetConf['numSamplingSufficientLocalPlanarPairwiseSurface']]
            idxA = tmp
            idxA3 = idxA[:, :, None].repeat(1, 1, 3)
            batch_thgpu['sufficientLocalPlanarPairwiseSurfaceAPCxyzWorld' + _dataset] = torch.gather(
                batch_thgpu['finalFullsurfacePCxyzWorld' + _dataset], dim=1, index=idxA3)
            batch_thgpu['sufficientLocalPlanarPairwiseSurfaceAPCnormalWorld' + _dataset] = torch.gather(
                batch_thgpu['finalFullsurfacePCnormalWorld' + _dataset], dim=1, index=idxA3)
            batch_thgpu['sufficientLocalPlanarPairwiseSurfaceAPCmaskfloat' + _dataset] = torch.ones(
                (B, datasetConf['numSamplingSufficientLocalPlanarPairwiseSurface']),
                dtype=torch.float32, device=dot.device)
            batch_thgpu['sufficientLocalPlanarPairwiseSurfaceAPCnnDot' + _dataset] = torch.gather(
                dot, dim=1, index=idxA)  # debugging purpose
            batch_thgpu['sufficientLocalPlanarPairwiseSurfaceAPCnnIdx' + _dataset] = torch.gather(
                nnIdx, dim=1, index=idxA)  # debugging purpose
            idxB = torch.gather(nnIdx, dim=1, index=tmp)
            idxB3 = idxB[:, :, None].repeat(1, 1, 3)
            batch_thgpu['sufficientLocalPlanarPairwiseSurfaceBPCxyzWorld' + _dataset] = torch.gather(
                batch_thgpu['finalFullsurfacePCxyzWorld' + _dataset], dim=1, index=idxB3)
            batch_thgpu['sufficientLocalPlanarPairwiseSurfaceBPCnormalWorld' + _dataset] = torch.gather(
                batch_thgpu['finalFullsurfacePCnormalWorld' + _dataset], dim=1, index=idxB3)
            batch_thgpu['sufficientLocalPlanarPairwiseSurfaceBPCmaskfloat' + _dataset] = torch.ones(
                (B, datasetConf['numSamplingSufficientLocalPlanarPairwiseSurface']),
                dtype=torch.float32, device=dot.device)
            del dot, nnIdx, tmp, idxA, idxA3, idxB, idxB3

        if 'nearsurface' in samplingMethodList:
            d = torch.rand(B, datasetConf['numSamplingSufficientNearsurface'], device=cudaDevice)
            d = (2 * d - 1) * datasetConf['nearsurfaceDeltaRange']
            batch_thgpu['sufficientNearsurfacePCxyzWorld' + _dataset] = \
                batch_thgpu['finalFullsurfacePCxyzWorld' + _dataset][
                :, -datasetConf['numSamplingSufficientNearsurface']:] + \
                d[:, :, None] * batch_thgpu['finalFullsurfacePCnormalWorld' + _dataset][
                                :, -datasetConf['numSamplingSufficientNearsurface']:]
            batch_thgpu['sufficientNearsurfacePCoccfloat' + _dataset] = (d > 0).float()
            batch_thgpu['sufficientNearsurfacePCsemantic' + _dataset] = (
                (batch_thgpu['finalFullsurfacePCsemantic' + _dataset][
                    :, -datasetConf['numSamplingSufficientNearsurface']:
                ])
                * (d <= 0)
            ).int()

            batch_thgpu['sufficientNearsurfacePCmaskfloat' + _dataset] = torch.ones_like(d)
            batch_thgpu['sufficientNearsurfacePCsdf' + _dataset] = d

        if 'selfnonsurface' in samplingMethodList:
            t = torch.rand(
                B, datasetConf['numSamplingSufficientSelfnonsurface'], 3, device=cudaDevice)
            t = t * (maxBoundWorld_thgpu - minBoundWorld_thgpu)[:, None, :] + \
                minBoundWorld_thgpu[:, None, :]
            tmp = knn_points(
                t, batch_thgpu['finalFullsurfacePCxyzWorld' + _dataset],
                K=1, return_nn=False, return_sorted=False,
            )
            dists = tmp.dists[:, :, 0] ** 0.5
            batch_thgpu['sufficientSelfnonsurfacePCxyzWorld' + _dataset] = t
            batch_thgpu['sufficientSelfnonsurfacePCmaskfloat' + _dataset] = torch.ones_like(dists)
            batch_thgpu['sufficientSelfnonsurfacePCmaskfloat' + _dataset][
                dists < datasetConf['selfnonsurfaceDeltaRange']] = -2
            batch_thgpu['sufficientSelfnonsurfacePCsdfabs' + _dataset] = dists

        # xyzCam and xyzCamPersp
        for samplingMethod in samplingMethodList:
            pcNameList = None
            if samplingMethod in ['fullsurface']:
                continue
            elif samplingMethod in ['nearsurface', 'selfnonsurface', 'unarysurface']:
                pcNameList = [samplingMethod]
            elif samplingMethod in ['localEdgePairwiseSurface', 'localPlanarPairwiseSurface']:
                pcNameList = [samplingMethod + x for x in ['A', 'B']]
            else:
                raise NotImplementedError('Unknown samplingMethod: %s' % samplingMethod)

            indexing_sufficient_to_final = None

            for pcName in pcNameList:
                camR = batch_thgpu['cam' + _dataset][:, :3, :3]
                camT = batch_thgpu['cam' + _dataset][:, :3, 3]
                xyzWorld = batch_thgpu['sufficient%sPCxyzWorld' % bt(pcName) + _dataset]
                xyzCam = torch.matmul(xyzWorld, camR.permute(0, 2, 1)) + camT[:, None, :]
                xyzCamPersp = torch.stack([
                    batch_thgpu['fScaleWidth' + _dataset][:, None] * torch.div(
                        xyzCam[:, :, 0], torch.clamp(xyzCam[:, :, 2], min=datasetConf['zNear'])),
                    batch_thgpu['fScaleHeight' + _dataset][:, None] * torch.div(
                        xyzCam[:, :, 1], torch.clamp(xyzCam[:, :, 2], min=datasetConf['zNear'])),
                    xyzCam[:, :, 2],
                ], 2)
                maskfloat = batch_thgpu['sufficient%sPCmaskfloat' % bt(pcName) + _dataset]
                maskfloat[
                    ((xyzCamPersp[:, :, 2] <= 0) |
                     (xyzCamPersp[:, :, 0] <= -1) | (xyzCamPersp[:, :, 0] >= 1) |
                     (xyzCamPersp[:, :, 1] <= -1) | (xyzCamPersp[:, :, 1] >= 1) | # |
                     torch.any(xyzWorld[:, :, :] <= minBoundWorld_thgpu[:, None, :], dim=2) |
                     torch.any(xyzWorld[:, :, :] >= maxBoundWorld_thgpu[:, None, :], dim=2)
                     # torch.any(xyzCam[:, :, :] <= minBoundCam_thgpu[:, None, :], dim=2) |
                     # torch.any(xyzCam[:, :, :] >= maxBoundCam_thgpu[:, None, :], dim=2)
                    ) &
                    (maskfloat == 1)
                    ] = -1.

                batch_thgpu['sufficient%sPCxyzCam' % bt(pcName) + _dataset] = xyzCam
                batch_thgpu['sufficient%sPCxyzCamPersp' % bt(pcName) + _dataset] = xyzCamPersp
                batch_thgpu['sufficient%sPCmaskfloat' % bt(pcName) + _dataset] = maskfloat

                if (('sufficient%sPCnormalWorld' % bt(pcName)) + _dataset) in batch_thgpu.keys():
                    batch_thgpu['sufficient%sPCnormalCam' % bt(pcName) + _dataset] = torch.matmul(
                        batch_thgpu['sufficient%sPCnormalWorld' % bt(pcName) + _dataset],
                        camR.permute(0, 2, 1))

                if (('sufficient%sPCnormalWorld' % bt(pcName)) + _dataset) in batch_thgpu.keys():
                    batch_thgpu['sufficient%sPCnormalCam' % bt(pcName) + _dataset] = torch.matmul(
                        batch_thgpu['sufficient%sPCnormalWorld' % bt(pcName) + _dataset],
                        camR.permute(0, 2, 1))

                # from sufficient to final
                #   flagging
                tmp = batch_thgpu['sufficient%sPCmaskfloat' % bt(pcName) + _dataset] > 0
                batch_thgpu['statMinMaskFloatHm%s' % bt(pcName)] = \
                    torch.tensor(
                        int(tmp.sum(1).min()), dtype=torch.int32,
                        device=batch_thgpu['sufficient%sPCmaskfloat' % bt(pcName) + _dataset].device
                    )
                batch_thgpu['statNumPerProcessMaskFloatZeroQueryHm%s' % bt(pcName)] = \
                    torch.tensor(
                        int((tmp.sum(1) < datasetConf['numSamplingFinal%s' % bt(samplingMethod)]).sum()),
                        dtype=torch.int32,
                        device=batch_thgpu['sufficient%sPCmaskfloat' % bt(pcName) + _dataset].device
                    )
                if indexing_sufficient_to_final is None:
                    tmp = torch.argsort(tmp.int(), dim=1, descending=True)[
                            :, :datasetConf['numSamplingFinal%s' % bt(samplingMethod)]]
                    indexing_sufficient_to_final = tmp
                else:
                    tmp = indexing_sufficient_to_final
                tmp3 = tmp[:, :, None].repeat(1, 1, 3)
                for k in ['xyzWorld', 'xyzCam', 'xyzCamPersp', 'normalCam']:
                    if ('sufficient%sPC%s' % (bt(pcName), k) + _dataset) in batch_thgpu.keys():
                        batch_thgpu['final%sPC%s' % (bt(pcName), k) + _dataset] = \
                            torch.gather(
                                batch_thgpu['sufficient%sPC%s' % (bt(pcName), k) + _dataset], dim=1, index=tmp3)
                for k in ['occfloat', 'sdf', 'semantic', 'nnIdx', 'nnDot']:
                    if ('sufficient%sPC%s' % (bt(pcName), k) + _dataset) in batch_thgpu.keys():
                        batch_thgpu['final%sPC%s' % (bt(pcName), k) + _dataset] = \
                            torch.gather(
                                batch_thgpu['sufficient%sPC%s' % (bt(pcName), k) + _dataset], dim=1, index=tmp)
                #   checking
                maskfloatFinal = torch.gather(
                    batch_thgpu['sufficient%sPCmaskfloat' % bt(pcName) + _dataset], dim=1, index=tmp)
                batch_thgpu['final%sPCmaskfloat' % bt(pcName) + _dataset] = maskfloatFinal

                # deleting
                del camR, camT, xyzWorld, xyzCam, xyzCamPersp, tmp, tmp3, maskfloatFinal

            del indexing_sufficient_to_final

        return batch_thgpu

    def batchPreprocessingTHGPU(self, batch_thcpu, batch_thgpu, **kwargs):
        iterCount = kwargs['iterCount']

        batch_thgpu = self.samplingBatchForScannetTHGPU(
            batch_thcpu,
            batch_thgpu,
            datasetObj=self.datasetObjDict['scannetGivenRender'],
            cudaDevice=self.cudaDeviceForAll,
            verbose=self.samplingVerbose,
            meta=self.meta,
            projRoot=self.projRoot,
            logDir=self.logDir,
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
    def doQueryPred0(cls, img0, **kwargs):
        queryPointCam0 = kwargs['queryPointCam0']
        queryPointCamPersp0 = kwargs['queryPointCamPersp0']
        models = kwargs['models']
        cudaDevice = kwargs['cudaDevice']
        batchSize = kwargs.get('batchSize', 100 ** 2)
        verboseBatchForwarding = kwargs.get('verboseBatchForwarding')

        cls.assertModelEvalMode(models)

        assert len(queryPointCam0.shape) == 2 and queryPointCam0.shape[1] == 3
        assert len(queryPointCamPersp0.shape) == 2 and queryPointCamPersp0.shape[1] == 3
        assert len(img0) == 3 and img0.shape[0] == 3

        encoder = models['encoder']
        decoder = models['decoder']
        positionalEncoder = models['positionalEncoder']
        c_dim_specs = models['meta']['c_dim_specs']

        nQuery = queryPointCam0.shape[0]
        assert nQuery == queryPointCamPersp0.shape[0]
        batchTot = int(math.ceil((float(nQuery) / batchSize)))
        ss = encoder(torch.from_numpy(img0[None, :, :, :]).contiguous().to(cudaDevice))

        predOccfloat0 = np.zeros((nQuery,), dtype=np.float32)
        for batchID in range(batchTot):
            if verboseBatchForwarding > 0 and (batchID < 5 or batchID % verboseBatchForwarding == 0):
                print('    Processing doQueryPred for batches %d / %d' % (batchID, batchTot))
            head = batchID * batchSize
            tail = min((batchID + 1) * batchSize, nQuery)

            pCam = torch.from_numpy(
                queryPointCam0[head:tail, :]
            ).to(cudaDevice)[None, :, :]
            pCamPersp = torch.from_numpy(
                queryPointCamPersp0[head:tail, :]
            ).to(cudaDevice)[None, :, :]

            tt = {}
            for j in [1, 2, 3, 4]:
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
            predOccfloat0[head:tail] = out.view(-1).detach().cpu().numpy()

        out = {'occfloat': predOccfloat0}
        out['depthPred'] = ss['x'].detach().cpu().numpy()[0, 0, :, :]
        return out

    @classmethod
    def doPred0(cls, bsv0, **kwargs):
        voxSize = kwargs['voxSize']
        models = kwargs['models']
        cudaDevice = kwargs['cudaDevice']
        verboseBatchForwarding = kwargs['verboseBatchForwarding']
        Trainer = kwargs['Trainer']
        useSurfaceOrDepth = kwargs['useSurfaceOrDepth']
        assert useSurfaceOrDepth in ['surface', 'depth', 'postprocess1']

        cls.assertModelEvalMode(models)

        tmp0 = gen_viewport_query_given_bound(
            Lx=voxSize, Ly=voxSize, Lz=voxSize,
            fScaleX=bsv0['fScaleWidth'], fScaleY=bsv0['fScaleHeight'],
            boundMin=bsv0['boundMinCam'], boundMax=bsv0['boundMaxCam'],
            zNear=bsv0['zNear'],
        )
        tmp1 = cls.doQueryPred0(
            bsv0['imgForUse'],
            queryPointCam0=tmp0['extractedXyzCam'],
            queryPointCamPersp0=tmp0['extractedXyzCamPersp'],
            models=models,
            cudaDevice=cudaDevice,
            verboseBatchForwarding=verboseBatchForwarding,
            Trainer=Trainer,
        )
        if useSurfaceOrDepth == 'surface':
            occfloat0 = np.ones((tmp0['Lx'] * tmp0['Ly'] * tmp0['Lz'],), dtype=np.float32)
            occfloat0[tmp0['flagQuery']] = tmp1['occfloat']
            occfloat0 = occfloat0.reshape((tmp0['Ly'], tmp0['Lx'], tmp0['Lz']))
            if occfloat0.min() > 0.5:
                occfloat0[0] = 0.
            if occfloat0.max() < 0.5:
                occfloat0[-1] = 1.
            predVertCam0, predFace0 = voxSdfSign2mesh_skmc(
                voxSdfSign=occfloat0, goxyz=tmp0['goxyzCam'], sCell=tmp0['sCell'],
            )
            bsv0['predVertCam'] = predVertCam0
            bsv0['predFace'] = predFace0
            bsv0['predDepth2'] = tmp1['depthPred']
        elif useSurfaceOrDepth == 'depth':
            tmp2 = depthMap2mesh(
                tmp1['depthPred'], float(bsv0['fScaleWidth']), float(bsv0['fScaleHeight']),
                cutLinkThre=0.1 * 2,
            )
            if tmp2['face0'].shape[0] == 0:
                tmp2['vertCam0'] = np.eye(3).astype(np.float32)
                tmp2['face0'] = np.array([[0, 1, 2]], dtype=np.int32)
            bsv0['predVertCam'] = tmp2['vertCam0']
            bsv0['predFace'] = tmp2['face0']
            bsv0['predDepth2'] = tmp1['depthPred']
        elif useSurfaceOrDepth == 'postprocess1':
            predVertCam0, predFace0 = cls.postprocess1(tmp0, tmp1, cudaDevice=cudaDevice)
            bsv0['predVertCam'] = predVertCam0
            bsv0['predFace'] = predFace0
            bsv0['predDepth2'] = tmp1['depthPred']
        else:
            raise NotImplementedError('Unknown useSurfaceOrDepth: %s' % useSurfaceOrDepth)
        bsv0['predMetricVertCam'] = bsv0['predVertCam'].copy()
        bsv0['predMetricFace'] = bsv0['predFace'].copy()
        cam0 = bsv0['cam']
        camR0, camT0 = cam0[:3, :3], cam0[:3, 3]
        bsv0['predMetricVertWorld'] = np.matmul(bsv0['predMetricVertCam'] - camT0[None, :], camR0) # camR0: transpose of inverse
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
    def forwardNetForScannet(cls, batch_thgpu, **kwargs):
        datasetConf = kwargs['datasetConf']
        models = kwargs['models']
        c_dim_specs = kwargs['c_dim_specs']
        iterCount = kwargs['iterCount']
        saveAnomalyFunc = kwargs['saveAnomalyFunc']

        dataset = datasetConf['dataset']
        _dataset = '_' + dataset

        encoder = models['encoder']
        decoder = models['decoder']
        positionalEncoder = models['positionalEncoder']
        # dgs2dLayer = models['dgs2dLayer']

        # network 1
        ss = encoder(batch_thgpu['imgForUse' + _dataset].contiguous())

        for pcName in ['nearsurface', 'selfnonsurface', 'unarysurface',
                        'localEdgePairwiseSurfaceA', 'localEdgePairwiseSurfaceB',
                        'localPlanarPairwiseSurfaceA', 'localPlanarPairwiseSurfaceB']:
            if ('final%sPCxyzCam' % bt(pcName) + _dataset) not in batch_thgpu.keys():
                continue
            pCam = batch_thgpu['final%sPCxyzCam' % bt(pcName) + _dataset].detach()
            '''  No need for scannet
            mask = batch_thgpu['final%sPCmaskfloat_hmRenderOm' % bt(pcName)]
            condition1 = pCam[:, :, 2] == 0
            condition2 = mask > 0
            if torch.any(condition1):
                if torch.any(condition1 & condition2):
                    print('[Anomaly] unmaskedZ0 in pcName: %s' % pcName)
                    saveAnomalyFunc(message='unmaskedZ0', iterCount=iterCount, batch_thgpu=batch_thgpu)
                    # won't stop here, but this will very likely to be a fatal.
                else:
                    t = pCam[:, :, 2]  # avoid NaN
                    t[t == 0] = datasetConf['zNear']
                    pCam[:, :, 2] = t
            '''
            if pcName in ['selfnonsurface', 'unarysurface',
                          'localEdgePairwiseSurfaceA', 'localEdgePairwiseSurfaceB',
                          'localPlanarPairwiseSurfaceA', 'localPlanarPairwiseSurfaceB']:
                pCam = pCam.requires_grad_(True)
            pCamPersp = torch.stack([
                batch_thgpu['fScaleWidth' + _dataset][:, None] * torch.div(
                    pCam[:, :, 0], torch.clamp(pCam[:, :, 2], min=datasetConf['zNear'])),
                batch_thgpu['fScaleHeight' + _dataset][:, None] * torch.div(
                    pCam[:, :, 1], torch.clamp(pCam[:, :, 2], min=datasetConf['zNear'])),
                pCam[:, :, 2],
            ], 2)

            if pcName in ['nearsurface']:
                tt = {}
                for j in [1, 2, 3, 4]:
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
                out = torch.clamp(out, min=1.e-5, max=1. - 1.e-5)
                batch_thgpu['final%sPCpredOccfloat' % bt(pcName) + _dataset] = \
                    out.view(pCam.shape[0], pCam.shape[1])
                del tt, phi, pe, lin, out
            elif pcName in ['selfnonsurface', 'unarysurface',
                            'localEdgePairwiseSurfaceA', 'localEdgePairwiseSurfaceB',
                            'localPlanarPairwiseSurfaceA', 'localPlanarPairwiseSurfaceB']:
                tt = {}
                for j in [1, 2, 3, 4]:
                    tt['t%d' % j] = pydgs_forwardValOnly(
                        ss['s%d' % j], pCamPersp[:, :, :2].contiguous(),
                    )
                phi = torch.cat([tt['t' + k[1:]] for k in c_dim_specs.keys()
                                 if k.startswith('s')], 2).contiguous()
                phi = phi.view(pCamPersp.shape[0] * pCamPersp.shape[1], -1).contiguous()
                pe = positionalEncoder(pCam.view(-1, 3).contiguous(),
                                       forwardMode='valOnly').contiguous()
                lin = decoder(pe, phi, forwardMode='valOnly')
                out = torch.sigmoid(lin)
                out = torch.clamp(out, min=1.e-5, max=1. - 1.e-5)
                batch_thgpu['final%sPCpredGradOccfloat' % bt(pcName) + _dataset] = \
                    torch.autograd.grad(out.sum(), [pCam], create_graph=True)[0]
                batch_thgpu['final%sPCpredOccfloat' % bt(pcName) + _dataset] = \
                    out.view(pCam.shape[0], pCam.shape[1])
                del tt, phi, pe, lin, out
                # tt4 = {}
                # for j in [1, 2, 3, 4]:
                #     tt4['t%d' % j] = dgs2dLayer(
                #         ss['s%d' % j], pCamPersp,
                #         batch_thgpu['fScaleWidth' + _dataset], batch_thgpu['fScaleHeight' + _dataset],
                #     ).permute(0, 3, 1, 2).contiguous()
                # phi4 = torch.cat([tt4['t' + k[1:]] for k in c_dim_specs.keys()
                #                   if k.startswith('s')], 2).contiguous()
                # phi4 = phi4.view(pCamPersp.shape[0] * pCamPersp.shape[1], -1, 4)
                # pe4 = positionalEncoder(
                #     pCam.view(-1, 3).contiguous(),
                #     forwardMode='gradTrack',
                # ).contiguous()
                # lin4 = decoder(pe4, phi4, forwardMode='gradTrack')
                # out0 = torch.sigmoid(lin4[..., :1])
                # out3 = out0 * (1 - out0) * lin4[..., 1:]
                # out4 = torch.cat([out0, out3], -1)
                # batch_thgpu['final%sPCpredOccfloat' % bt(pcName) + _dataset] = \
                #     out4[..., 0].view(pCam.shape[0], pCam.shape[1]).contiguous()
                # batch_thgpu['final%sPCpredGradOccfloat' % bt(pcName) + _dataset] = \
                #     out4[..., 1:].view(pCam.shape[0], pCam.shape[1], 3).contiguous()
            else:
                raise NotImplementedError('Unknown pcName: %s' % pcName)

        batch_thgpu['depthPred' + _dataset] = ss['x']

        return batch_thgpu

    @classmethod
    def forwardNetForOmni(cls, batch_thgpu, **kwargs):
        models = kwargs['models']
        dataset = kwargs['dataset']

        encoder = models['encoder']
        ss = encoder(batch_thgpu['rgb_%s' % dataset])
        batch_thgpu['depthPred_%s' % dataset] = ss['x']
        return batch_thgpu

    @staticmethod
    def make_valid_mask(mask_float, max_pool_size=4, return_small_mask=False):
        '''
            Creates a mask indicating the valid parts of the image(s).
            Enlargens masked area using a max pooling operation.

            Args:
                mask_float: A mask as loaded from the Taskonomy loader.
                max_pool_size: Parameter to choose how much to enlarge masked area.
                return_small_mask: Set to true to return mask for aggregated image
        '''
        if len(mask_float.shape) == 3:
            mask_float = mask_float.unsqueeze(axis=0)
        elif len(mask_float.shape) == 2:
            mask_float = mask_float.unsqueeze(axis=0).unsqueeze(axis=0)

        h, w = mask_float.shape[2], mask_float.shape[3]
        reshape_temp = len(mask_float.shape) == 5
        if reshape_temp:
            mask_float = rearrange(mask_float, 'b p c h w -> (b p) c h w')
        mask_float = 1 - mask_float
        mask_float = F.max_pool2d(mask_float, kernel_size=max_pool_size)
        # mask_float = F.interpolate(mask_float, (self.image_size, self.image_size), mode='nearest')
        mask_float = F.interpolate(mask_float, (h, w), mode='nearest')
        mask_valid = mask_float == 0
        if reshape_temp:
            raise NotImplementedError('This should never be reached.')
            mask_valid = rearrange(mask_valid, '(b p) c h w -> b p c h w', p=self.num_positive)

        return mask_valid

    @classmethod
    def forwardLoss(cls, batch_thgpu, **kwargs):
        wl = kwargs['wl']
        ifRequiresGrad = kwargs['ifRequiresGrad']

        iterCount = kwargs['iterCount']
        midas_loss_fn = kwargs['midas_loss_fn']
        vnl_loss_fn = kwargs['vnl_loss_fn']
        ifFinetunedFromOmnitools = kwargs['ifFinetunedFromOmnitools']

        iterThre = 15000

        if ('lossOmniSsi' in wl.keys() and wl['lossOmniSsi'] > 0) or \
                ('lossOmniReg' in wl.keys() and wl['lossOmniReg'] > 0):
            dataset = 'omnidataBerkeley'
            _, ssi_loss, reg_loss = midas_loss_fn(
                batch_thgpu['depthPred_%s' % dataset],
                batch_thgpu['depthZBuffer_%s' % dataset],
                cls.make_valid_mask(batch_thgpu['maskValid_%s' % dataset]),
            )
            batch_thgpu['lossOmniSsi'] = ssi_loss
            batch_thgpu['lossOmniReg'] = reg_loss \
                if (iterCount > iterThre) or (ifFinetunedFromOmnitools > 0) else 0.
            del dataset, ssi_loss, reg_loss

        if 'lossOmniVnl' in wl.keys() and wl['lossOmniVnl'] > 0:
            dataset = 'omnidataBerkeley'
            vnl_loss = vnl_loss_fn(
                batch_thgpu['depthPred_%s' % dataset], batch_thgpu['depthZBuffer_%s' % dataset])
            batch_thgpu['lossOmniVnl'] = vnl_loss \
                if (iterCount > iterThre) or (ifFinetunedFromOmnitools > 0) else 0.
            del dataset, vnl_loss

        # lossDepthRegL1
        if 'lossDepthRegL1' in wl.keys() and wl['lossDepthRegL1'] > 0:
            depthRaw = batch_thgpu['depthForUse_scannetGivenRender']
            depthPred = batch_thgpu['depthPred_scannetGivenRender']
            mask = torch.isfinite(depthRaw)
            batch_thgpu['lossDepthRegL1'] = (depthRaw[mask] - depthPred[mask]).abs().mean()
            del depthRaw, depthPred, mask

        # lossNearsurfaceClf
        if 'lossNearsurfaceClf' in wl.keys() and wl['lossNearsurfaceClf'] > 0:
            tmp = F.binary_cross_entropy(
                batch_thgpu['finalNearsurfacePCpredOccfloat_scannetGivenRender'],
                batch_thgpu['finalNearsurfacePCoccfloat_scannetGivenRender'],
                reduction='none',
            )
            batch_thgpu['lossNearsurfaceClf'] = (
                    tmp * (batch_thgpu['finalNearsurfacePCmaskfloat_scannetGivenRender'] > 0).float()
            ).mean()
            del tmp

        # lossSelfnonsurfaceGradOccfloat
        if ifRequiresGrad and 'lossSelfnonsurfaceGradOccfloat' in wl.keys() \
                and wl['lossSelfnonsurfaceGradOccfloat'] > 0:
            batch_thgpu['lossSelfnonsurfaceGradOccfloat'] = torch.abs(
                batch_thgpu['finalSelfnonsurfacePCpredGradOccfloat_scannetGivenRender'] *
                (batch_thgpu['finalSelfnonsurfacePCmaskfloat_scannetGivenRender'] > 0).float()[:, :, None]
            ).mean()

        # lossUnarysurfaceNormal
        if ifRequiresGrad and 'lossUnarysurfaceNormal' in wl.keys() and wl['lossUnarysurfaceNormal'] > 0:
            mask = (batch_thgpu['finalUnarysurfacePCmaskfloat_scannetGivenRender'] > 0)  # (B, Q)
            predGrad = batch_thgpu['finalUnarysurfacePCpredGradOccfloat_scannetGivenRender']  # (B, Q, 3)
            labelGrad = batch_thgpu['finalUnarysurfacePCnormalCam_scannetGivenRender']  # (B, Q, 3)
            dot = (predGrad * labelGrad).sum(2)  # (B, Q)
            predNorm = torch.clip(torch.norm(predGrad, dim=2, p=2), min=1.e-4)  # (B, Q)
            labelNorm = torch.clip(torch.norm(labelGrad, dim=2, p=2), min=1.e-4)  # (B, Q)
            delta = torch.abs(predGrad / predNorm[:, :, None] - labelGrad / labelNorm[:, :, None]).mean(2)
            batch_thgpu['lossUnarysurfaceNormal'] = delta[mask].mean()
            del mask, predGrad, labelGrad, dot, predNorm, labelNorm, delta

        # lossLocal[Edge|Planar]PairwiseSurfaceNormal
        for edgeOrPlanar in ['edge', 'planar']:
            if ifRequiresGrad and ('lossLocal%sPairwiseSurfaceNormal' % bt(edgeOrPlanar) in wl.keys()) and \
                    (wl['lossLocal%sPairwiseSurfaceNormal' % bt(edgeOrPlanar)] > 0):
                maskA = (batch_thgpu['finalLocal%sPairwiseSurfaceAPCmaskfloat_scannetGivenRender' % bt(edgeOrPlanar)] > 0)
                maskB = (batch_thgpu['finalLocal%sPairwiseSurfaceBPCmaskfloat_scannetGivenRender' % bt(edgeOrPlanar)] > 0)
                mask = maskA & maskB  # (B, Q)
                predAGrad = batch_thgpu['finalLocal%sPairwiseSurfaceAPCpredGradOccfloat_scannetGivenRender' % bt(edgeOrPlanar)]
                predANorm = torch.clip(torch.norm(predAGrad, dim=2, p=2), min=1.e-4)
                predBGrad = batch_thgpu['finalLocal%sPairwiseSurfaceBPCpredGradOccfloat_scannetGivenRender' % bt(edgeOrPlanar)]
                predBNorm = torch.clip(torch.norm(predBGrad, dim=2, p=2), min=1.e-4)
                labelAGrad = batch_thgpu['finalLocal%sPairwiseSurfaceAPCnormalCam_scannetGivenRender' % bt(edgeOrPlanar)]
                labelANorm = torch.clip(torch.norm(labelAGrad, dim=2, p=2), min=1.e-4)
                labelBGrad = batch_thgpu['finalLocal%sPairwiseSurfaceBPCnormalCam_scannetGivenRender' % bt(edgeOrPlanar)]
                labelBNorm = torch.clip(torch.norm(labelBGrad, dim=2, p=2), min=1.e-4)
                delta = torch.abs(
                    (predAGrad * predBGrad).sum(2) / predANorm / predBNorm -
                    (labelAGrad * labelBGrad).sum(2) / labelANorm / labelBNorm
                )
                batch_thgpu['lossLocal%sPairwiseSurfaceNormal' % bt(edgeOrPlanar)] = delta[mask].mean()
                del maskA, maskB, mask, predAGrad, predANorm, predBGrad, predBNorm
                del labelAGrad, labelANorm, labelBGrad, labelBNorm, delta

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

        self.assertModelsTrainingMode(self.models)

        config = self.config

        printInfo(self.rank, 'beforeForwardScannet')
        batch_thgpu = self.forwardNetForScannet(
            batch_thgpu,
            models=self.models,
            meta=self.meta,
            datasetConf=self.config.datasetConfDict['scannetGivenRender'],
            c_dim_specs=config.get('c_dim_specs', None),
            ifRequiresGrad=ifRequiresGrad,
            iterCount=iterCount,
            saveAnomalyFunc=self.saveAnomaly,
        )
        printInfo(self.rank, 'beforeForwardOmni')
        batch_thgpu = self.forwardNetForOmni(
            batch_thgpu,
            models=self.models,
            dataset='omnidataBerkeley',
        )

        # anomaly in batch_thgpu
        '''
        breakFlag = False
        for k in batch_thgpu.keys():
            if (type(batch_thgpu[k]) is torch.Tensor) and (k not in ['depthForUse_hmRenderOm']):
                if not torch.all(torch.isfinite(batch_thgpu[k])):
                    print('[Anomaly] nanDetectedInBatchThgpu: %s (rank: %d)' % (k, self.rank))
                    if not breakFlag:
                        self.saveAnomaly(message='nanDetectedInBatchThgpu', iterCount=iterCount, batch_thgpu=batch_thgpu)
                        breakFlag = True
        # Let it continue to run. Sometimes having NaN is OK, as it will be masked out.
        '''

        printInfo(self.rank, 'beforeForwardLoss')
        batch_thgpu = self.forwardLoss(
            batch_thgpu,
            wl=config.wl,
            iterCount=iterCount,
            ifRequiresGrad=ifRequiresGrad,
            midas_loss_fn=self.midas_loss,
            vnl_loss_fn=self.vnl_loss,
            ifFinetunedFromOmnitools=config.ifFinetunedFromOmnitools,
        )
        printInfo(self.rank, 'beforeBackward')
        if ifAllowBackward:
            self.backwardLoss(
                batch_thgpu, iterCount=iterCount, optimizerModels=self.optimizerModels,
            )
        printInfo(self.rank, 'afterBackward')

        # anomaly in models
        '''
        breakFlag = False
        for modelName, model in self.models.items():
            if (modelName == 'meta') or (modelName in self.models['meta']['nonLearningModelNameList']):
                continue
            if self.numMpProcess <= 0:
                module = model
            else:
                module = model.module
            for k, v in module.state_dict().items():
                if not torch.all(torch.isfinite(v)):
                    print('[Anomaly] nanDetectedInModels: %s %s (rank: %d)' % (modelName, k, self.rank))
                    if not breakFlag:
                        self.saveAnomaly(message='nanDetectedInModels', iterCount=iterCount, batch_thgpu=batch_thgpu)
                        breakFlag = True
        if breakFlag:  # This breakFlag will definitely stop the training process
            import ipdb
            ipdb.set_trace()
            print(1 + 1)
        '''

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
        pyrenderManager = kwargs['pyrenderManager']

        config = self.config
        S = config.S
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
            power10 = min(4, power10)
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

            ifMonitorImmediateFlow = iterCount == self.monitorMode

            if S.startswith('Sdummy'):
                ifStore = False
                ifBackupTheLatestModels = False
                ifMonitorDump = False
                ifMonitorImmediateFlow = False
                ifMonitorTrain = False
                ifMonitorVal = False
                ifSaveToFinishedIter = False

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
                # log virtual env name
                self.logger.info('    [VirtualEnv] %s' % sys.executable)

                # print(
                #     '---------- Iter %d Training: Host %s GPU %s CPU %d '
                #     'Rank %d NumMpProcess %d %s %s %s %s ---------' %
                #     (iterCount, gethostname(),
                #      os.environ['CUDA_VISIBLE_DEVICES'], os.getpid(),
                #      self.rank, self.numMpProcess, config.P,
                #      config.D, config.S,
                #      config.R))
                # print('    [TimeStamp] timeStamp Iter%d: ' % iterCount +
                #                  time.strftime('%m/%d/%y %H:%M:%S', time.localtime()))
                # # log virtual env name
                # print('    [VirtualEnv] %s' % sys.executable)


            # -------------------------------- batch data loading -------------------------------- #
            t = time.time()

            batch_thcpu, batch_thgpu = self.stepBatch(iterCount=iterCount)

            if ifPrint and ifTracked:
                self.logger.info('    [Timer] dataLoading Iter%d: %.3f seconds' % (iterCount, time.time() - t))

            # ----------------------------------- Preprocessing ----------------------------------- #
            t = time.time()
            printInfo(self.rank, 'Before batchPreprocessingTHGPU')
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

            # monitorImmediateFlow
            if ifMonitorImmediateFlow:
                if self.numMpProcess > 0:
                    dist.barrier()

                with torch.no_grad():
                    self.setModelsEvalMode(self.models)
                    if self.numMpProcess > 0:
                        dist.barrier()
                    batch_vis = castAnything(batch_thgpu, 'thgpu2np')
                    ben = self.stepMonitorImmediateFlow(
                        batch_vis, iterCount=iterCount,
                        ifMonitorDump=True,  # ifMonitorDump,
                        pyrenderManager=pyrenderManager,
                    )
                    '''
                    for k in ben.keys():
                        batch_thgpu['statTrain' + bt(k)] = \
                            torch.nanmean(torch.from_numpy(ben[k]).to(self.cudaDeviceForAll))
                    '''
                    if self.numMpProcess > 0:
                        dist.barrier()
                    self.setModelsTrainingMode(self.models)

                if self.numMpProcess > 0:
                    dist.barrier()

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
                                                pyrenderManager=pyrenderManager,
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
                                              pyrenderManager=pyrenderManager,
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


