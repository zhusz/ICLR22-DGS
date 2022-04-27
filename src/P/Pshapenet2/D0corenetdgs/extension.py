# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


bt = lambda s: s[0].upper() + s[1:]


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
        import torch
        from torch.nn import functional as F
        from .. import resnet50
        from ..decoder import Decoder
        from codes_py.toolbox_3D.dgs_wrapper_v1 import DGS3DLayer
        import io
        from .. import file_system as fs
        from ..trainer import Trainer
        import pickle
        class DTrainer(Trainer):
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

                dgs3dLayer = DGS3DLayer(sizeX=1., sizeY=1., sizeZ=1.)

                meta = {
                    'nonLearningModelNameList': ['dgs3dLayer'],
                }
                self.models = dict()
                self.models['meta'] = meta
                self.models['encoder'] = encoder.to(self.cudaDeviceForAll)
                self.models['networkTwo'] = networkTwo.to(self.cudaDeviceForAll)
                self.models['dgs3dLayer'] = dgs3dLayer.to(self.cudaDeviceForAll)

            @classmethod
            def forwardNetForCorenet(cls, batch_thgpu, **kwargs):
                models = kwargs['models']
                meta = kwargs['meta']

                corenetCamcubeVoxXyzCamPerspZYX_thgpuDict = \
                    meta['corenetCamcubeVoxXyzCamPerspZYX_thgpuDict']
                f0 = meta['f0']

                encoder = models['encoder']
                networkTwo = models['networkTwo']
                dgs3dLayer = models['dgs3dLayer']

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
                y6 = torch.sigmoid(yy['y6'])
                batch_thgpu['corenetCamcubePredBisemZYX_corenetChoySingleRendering'] = yy['y6'][:, 0, :, :, :]

                for samplingMethod in ['nearsurfacemass', 'nearsurfaceair', 'nonsurface']:
                    pCam = batch_thgpu['final%sPCxyzCam_corenetChoySingleRendering' % bt(samplingMethod)]
                    grid = torch.stack([
                        pCam[:, :, 0] * 2.,  # x from [-0.5, 0.5] to [-1, 1]
                        pCam[:, :, 1] * 2.,  # y from [-0.5, 0.5] to [-1, 1]
                        (pCam[:, :, 2] - f0 / 2. - 0.5) * 2.,  # z from [f0/2, 1 + f0/2] to [-1, 1]
                    ], 2)

                    if samplingMethod in ['nearsurfacemass', 'nearsurfaceair']:
                        z6 = F.grid_sample(
                            y6, grid[:, :, None, None, :],
                            mode='bilinear', padding_mode='border', align_corners=False,
                        )[:, 0, :, 0, 0]
                        batch_thgpu['final%sPCpredBisem_corenetChoySingleRendering' % bt(samplingMethod)] = z6
                    elif samplingMethod in ['nonsurface']:
                        z6 = dgs3dLayer(y6, grid)[:, 0, :, :].permute(0, 2, 1)
                        batch_thgpu['final%sPCpredBisem_corenetChoySingleRendering' % bt(samplingMethod)] = z6[:, :, 0]
                        batch_thgpu['final%sPCpredGradBisem_corenetChoySingleRendering' % bt(samplingMethod)] = z6[:, :, 1:]
                    else:
                        raise NotImplementedError('Unknown samplingMethod: %s' % samplingMethod)

                return batch_thgpu

            @classmethod
            def forwardLoss(cls, batch_thgpu, **kwargs):
                wl = kwargs['wl']

                # lossNearsurface[mass|air]
                for samplingMethod in ['nearsurfacemass', 'nearsurfaceair']:
                    if wl['loss%sClf' % bt(samplingMethod)] > 0:
                        pred = batch_thgpu['final%sPCpredBisem_corenetChoySingleRendering' % bt(samplingMethod)]
                        label = batch_thgpu['final%sPClabelBisem_corenetChoySingleRendering' % bt(samplingMethod)]

                        pred = torch.clamp(pred, min=0, max=1. - 1e-6 if samplingMethod == 'nearsurfaceair' else 1)
                        # Note F.grid_sample sometimes outputs slightly >1 numbers, even if input.max() == 1
                        # (due to numerical issues).
                        # >1 numbers of pred would result in errors in the following F.binary_cross_entropy function
                        # for samples where label==0 (nearsurfaceair)
                        # This happens once roughly every 100K iterations.

                        tmp = F.binary_cross_entropy(pred, label, reduction='none')
                        batch_thgpu['loss%sClf' % bt(samplingMethod)] = tmp.mean()
                        del pred, label, tmp
                del samplingMethod

                # lossNonsurfaceClfGradZero
                if wl['lossNonsurfaceClfGradZero'] > 0:
                    predGrad = batch_thgpu['final%sPCpredGradBisem_corenetChoySingleRendering' % bt('nonsurface')]
                    batch_thgpu['lossNonsurfaceClfGradZero'] = torch.abs(predGrad).mean()
                    del predGrad

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
        exportedClasses['DTrainer'] = DTrainer

    return exportedClasses

