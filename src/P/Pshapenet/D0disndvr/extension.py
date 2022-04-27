# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


bt = lambda s: s[0].upper() + s[1:]


def returnExportedClasses(wishedClassNameList):

    _testSuiteD = os.path.basename(os.path.dirname(__file__))

    exportedClasses = {}

    if wishedClassNameList is None or \
            'DCorenetChoySingleRenderingDataset' in wishedClassNameList:
        from ..dataset import CorenetChoySingleRenderingDataset
        class DCorenetChoySingleRenderingDataset(CorenetChoySingleRenderingDataset):
            pass
        exportedClasses['DCorenetChoySingleRenderingDataset'] = \
            DCorenetChoySingleRenderingDataset

    if wishedClassNameList is None or 'DTrainer' in wishedClassNameList:
        import torch
        from torch.nn import functional as F
        from torch.nn.functional import grid_sample
        import io
        from .. import file_system as fs
        import numpy as np
        from codes_py.toolbox_3D.rotations_v1 import camSys2CamPerspSysTHGPU
        from ..resnetEncoderForDisn import ResnetEncoder
        from ..resnetfc import ResNetFC
        from ..positionalEncoding import PositionalEncoding
        from ..trainer import Trainer
        class DTrainer(Trainer):
            def _netConstruct(self, **kwargs):
                config = self.config
                self.logger.info('[Trainer] MetaModelLoading - _netConstruct')

                encoder = ResnetEncoder(
                    requires_grad=True, requires_gaussian_blur=False, encoderTag=config.encoderTag,
                    ifNeedInternalRgbNormalization=True)

                networkTwo = ResNetFC(
                    d_in=39, d_out=1, n_blocks=5, d_latent=encoder.c_dim, d_hidden=512, beta=0.0,
                    combine_layer=3
                )
                positionalEncoder = PositionalEncoding(
                    num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True
                )

                meta = {
                    'nonLearningModelNameList': ['positionalEncoder'],
                }
                self.models = dict()
                self.models['meta'] = meta
                self.models['encoder'] = encoder.to(self.cudaDeviceForAll)
                self.models['networkTwo'] = networkTwo.to(self.cudaDeviceForAll)
                self.models['positionalEncoder'] = positionalEncoder.to(self.cudaDeviceForAll)

            @classmethod
            def _samplingFromNetwork2ToNetwork3(cls, yy, pointPCxyzCam, f0, **kwargs):
                raise NotImplementedError('This method should not be called')

            @classmethod
            def forwardNet(cls, batch_thgpu, **kwargs):
                models = kwargs['models']

                encoder = models['encoder']
                networkTwo = models['networkTwo']
                positionalEncoder = models['positionalEncoder']

                # network 1
                c_list = encoder(batch_thgpu['img'].contiguous())

                for samplingMethod in ['nearsurfacemass', 'nearsurfaceair', 'nonsurface']:
                    pCam = batch_thgpu['final%sPCxyzCam' % bt(samplingMethod)]
                    pCamPersp = camSys2CamPerspSysTHGPU(
                        pCam,
                        batch_thgpu['focalLengthWidth'], batch_thgpu['focalLengthHeight'],
                        batch_thgpu['winWidth'], batch_thgpu['winHeight'])
                    phi = torch.cat([
                        grid_sample(
                            c, pCamPersp[:, :, None, :2], mode='bilinear', padding_mode='reflection',
                            align_corners=False,
                        )
                        for c in c_list
                    ], 1)[:, :, :, 0].permute(0, 2, 1).contiguous().view(
                        pCamPersp.shape[0] * pCamPersp.shape[1], encoder.c_dim)
                    out = torch.sigmoid(networkTwo(positionalEncoder(pCam.view(-1, 3).contiguous(), forwardMode='valOnly').contiguous(), phi, forwardMode='valOnly'))
                    batch_thgpu['final%sPCpredBisem' % bt(samplingMethod)] = \
                        out.view(pCam.shape[0], pCam.shape[1])
                return batch_thgpu

            @classmethod
            def forwardLoss(cls, batch_thgpu, **kwargs):
                wl = kwargs['wl']

                # lossNearsurface[mass|air]
                for samplingMethod in ['nearsurfacemass', 'nearsurfaceair', 'nonsurface']:
                    if wl['loss%sClf' % bt(samplingMethod)] > 0:
                        pred = batch_thgpu['final%sPCpredBisem' % bt(samplingMethod)]
                        label = batch_thgpu['final%sPClabelBisem' % bt(samplingMethod)]
                        tmp = F.binary_cross_entropy(pred, label, reduction='none')
                        batch_thgpu['loss%sClf' % bt(samplingMethod)] = tmp.mean()
                        del pred, label, tmp
                del samplingMethod

                # weighted
                for k in batch_thgpu.keys():
                    if k.startswith('loss'):
                        batch_thgpu[k] *= wl[k]

                # loss total
                batch_thgpu['loss'] = sum([batch_thgpu[k] for k in batch_thgpu.keys()
                                           if k.startswith('loss')])

                return batch_thgpu

        exportedClasses['DTrainer'] = DTrainer

    if wishedClassNameList is None or 'DVisualizer' in wishedClassNameList:
        import torch
        from torch.nn.functional import grid_sample
        from ..visual import Visualizer
        import numpy as np
        import math
        class DVisualizer(Visualizer):
            @staticmethod
            def doQueryPred0(img0, queryPointCam0, queryPointCamPersp0, **kwargs):

                meta = kwargs['meta']
                models = kwargs['models']
                cudaDevice = kwargs['cudaDevice']
                batchSize = kwargs.get('batchSize', 100 ** 2)
                verboseBatchForwarding = kwargs.get('verboseBatchForwarding', 1)

                assert len(queryPointCam0.shape) == 2 and queryPointCam0.shape[1] == 3
                assert len(queryPointCamPersp0.shape) == 2 and queryPointCamPersp0.shape[1] == 3
                assert len(img0) == 3 and img0.shape[0] == 3

                encoder = models['encoder']
                networkTwo = models['networkTwo']
                positionalEncoder = models['positionalEncoder']
                encoder.eval()
                networkTwo.eval()
                positionalEncoder.eval()

                nQuery = queryPointCam0.shape[0]
                assert nQuery == queryPointCamPersp0.shape[0]
                batchTot = int(math.ceil((float(nQuery) / batchSize)))

                # encoder (Network One)
                c_list = encoder(
                    torch.from_numpy(img0[None, :, :, :]).float().contiguous().to(cudaDevice)
                )

                predBisem0 = np.zeros((nQuery,), dtype=np.float32)
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

                    phi = torch.cat([
                        grid_sample(
                            c, pCamPersp[:, :, None, :2], mode='bilinear', padding_mode='reflection',
                            align_corners=False,
                        )
                        for c in c_list
                    ], 1)[:, :, :, 0].permute(0, 2, 1).contiguous().view(
                        pCamPersp.shape[0] * pCamPersp.shape[1], encoder.c_dim)
                    out = torch.sigmoid(networkTwo(positionalEncoder(pCam.view(-1, 3).contiguous(), forwardMode='valOnly').contiguous(), phi, forwardMode='valOnly'))[:, 0]
                    predBisem0[head:tail] = out.detach().cpu().numpy()

                predOccfloat0 = 1. - predBisem0
                return {'occfloat': predOccfloat0, 'bisem': predBisem0}
        exportedClasses['DVisualizer'] = DVisualizer

    return exportedClasses

