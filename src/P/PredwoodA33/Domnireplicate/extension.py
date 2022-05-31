import os


bt = lambda s: s[0].upper() + s[1:]


def returnExportedClasses(wishedClassNameList):  # To avoid import unnecessary class of different envs

    _testSuiteD = os.path.basename(os.path.dirname(__file__))

    exportedClasses = {}

    if wishedClassNameList is None or 'DScannetGivenRenderDataset' in wishedClassNameList:
        from ..dataset import PScannetGivenRenderDataset
        class DScannetGivenRenderDataset(PScannetGivenRenderDataset):
            pass
        exportedClasses['DScannetGivenRenderDataset'] = DScannetGivenRenderDataset

    if wishedClassNameList is None or 'DTrainer' in wishedClassNameList:
        import torch
        import torch.distributed as dist
        from ..trainer import Trainer
        from ..midas.dpt_depth import DPTDepthModel
        import time
        import math
        from socket import gethostname
        from codes_py.toolbox_framework.framework_util_v4 import \
            checkPDSRLogDirNew, castAnything, probe_load_network, load_network, save_network, \
            bsv02bsv, mergeFromAnotherBsv0, bsv2bsv0, constructInitialBatchStepVis0, \
            batchExtractParticularDataset
        class DTrainer(Trainer):
            def _netConstruct(self):
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

                meta = {
                    'nonLearningModelNameList': [],
                }
                self.models = dict()
                self.models['encoder'] = encoder.to(self.cudaDeviceForAll)
                self.models['meta'] = meta

            def stepMonitorTrain(self, batch_vis, **kwargs):
                return {}

            def stepMonitorVal(self, **kwargs):
                return {}

            @classmethod
            def doQueryPred0(cls, img0, **kwargs):
                models = kwargs['models']
                cudaDevice = kwargs['cudaDevice']

                cls.assertModelEvalMode(models)

                assert len(img0) == 3 and img0.shape[0] == 3

                encoder = models['encoder']

                ss = encoder(torch.from_numpy(img0[None, :, :, :]).contiguous().to(cudaDevice))

                out = {}
                out['depthPred'] = ss['x'].detach().cpu().numpy()[0, 0, :, :]
                return out

            def forwardBackwardUpdate(self, batch_thgpu, **kwargs):
                ifAllowBackward = kwargs['ifAllowBackward']
                iterCount = kwargs['iterCount']
                ifRequiresGrad = kwargs['ifRequiresGrad']

                config = self.config

                batch_thgpu = self.forwardNetForOmni(
                    batch_thgpu,
                    models=self.models,
                    dataset='omnidataBerkeley',
                )
                batch_thgpu = self.forwardLoss(
                    batch_thgpu,
                    wl=config.wl,
                    iterCount=iterCount,
                    ifRequiresGrad=ifRequiresGrad,
                    midas_loss_fn=self.midas_loss,
                    vnl_loss_fn=self.vnl_loss,
                    ifFinetunedFromOmnitools=config.ifFinetunedFromOmnitools,
                )
                if ifAllowBackward:
                    self.backwardLoss(
                        batch_thgpu, iterCount=iterCount, optimizerModels=self.optimizerModels,
                    )
                return batch_thgpu

            def batchPreprocessingTHGPU(self, batch_thcpu, batch_thgpu, **kwargs):
                return batch_thgpu

        exportedClasses['DTrainer'] = DTrainer

    return exportedClasses

