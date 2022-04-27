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

    if wishedClassNameList is None or 'DScannetGivenRenderDataset' in wishedClassNameList:
        from ..dataset import PScannetGivenRenderDataset
        class DScannetGivenRenderDataset(PScannetGivenRenderDataset):
            pass
        exportedClasses['DScannetGivenRenderDataset'] = DScannetGivenRenderDataset

    if wishedClassNameList is None or 'DTrainer' in wishedClassNameList:
        import torch
        from torch.nn.functional import grid_sample
        from ..trainer import Trainer
        class DTrainer(Trainer):
            @classmethod
            def _samplingFromNetwork1_DISN(cls, ss, pointPCxyzCamPersp, **kwargs):
                B = ss['s32'].shape[0]
                assert len(pointPCxyzCamPersp.shape) == 3 and pointPCxyzCamPersp.shape[0] == B and \
                       pointPCxyzCamPersp.shape[2] == 3
                g = lambda s: grid_sample(
                    s, pointPCxyzCamPersp[:, :, None, :2],
                    mode='bilinear', padding_mode='border', align_corners=False
                )[:, :, :, 0].permute(0, 2, 1).contiguous()
                tt = {}
                for j in [32, 16, 8, 4, 2]:
                    if 's%d' % j in ss.keys():
                        tt['t%d' % j] = g(ss['s%d' % j])
                return tt

            @classmethod
            def forwardNet(cls, batch_thgpu, **kwargs):
                datasetMetaConf = kwargs['datasetMetaConf']
                models = kwargs['models']
                c_dim_specs = kwargs['c_dim_specs']
                ifRequiresGrad = kwargs['ifRequiresGrad']

                encoder = models['encoder']
                decoder = models['decoder']
                positionalEncoder = models['positionalEncoder']

                # network 1
                ss = encoder(batch_thgpu['imgForUse'].contiguous())

                for samplingMethod in ['nearsurface', 'selfnonsurface']:
                    pCam = batch_thgpu['final%sPCxyzCam' % bt(samplingMethod)].requires_grad_(ifRequiresGrad)
                    pCamPersp = torch.stack([
                        datasetMetaConf['fScaleWidth'] * torch.div(
                            pCam[:, :, 0], torch.clamp(pCam[:, :, 2], min=datasetMetaConf['zNear'])),
                        datasetMetaConf['fScaleHeight'] * torch.div(
                            pCam[:, :, 1], torch.clamp(pCam[:, :, 2], min=datasetMetaConf['zNear'])),
                        pCam[:, :, 2],
                    ], 2)

                    # sampling from network 1
                    tt = cls._samplingFromNetwork1_DISN(ss, pCamPersp)

                    out = torch.sigmoid(decoder(
                        positionalEncoder(pCam.view(-1, 3).contiguous(),
                                          forwardMode='valOnly').contiguous(),
                        torch.cat([tt['t' + k[1:]] for k in c_dim_specs.keys()
                                   if k.startswith('s')], 2).contiguous(
                        ).view(pCamPersp.shape[0] * pCamPersp.shape[1], -1).contiguous(),
                        forwardMode='valOnly',
                    ))
                    batch_thgpu['final%sPCpredOccfloat' % bt(samplingMethod)] = out.view(pCam.shape[0], pCam.shape[1])
                    if samplingMethod in ['selfnonsurface'] and ifRequiresGrad:
                        grad = torch.autograd.grad(out.sum(), [pCam], create_graph=True)[0]
                        batch_thgpu['final%sPCpredGradOccfloat' % bt(samplingMethod)] = grad

                        # RuntimeError will be raised here:
                        # RuntimeError: derivative for aten::grid_sampler_2d_backward is not implemented

                        del grad
                    del pCam, pCamPersp, tt, out

                batch_thgpu['depthPred'] = ss['x']

                return batch_thgpu

        exportedClasses['DTrainer'] = DTrainer

    if wishedClassNameList is None or 'DVisualizer' in wishedClassNameList:
        from ..visual import Visualizer
        class DVisualizer(Visualizer):
            pass
        exportedClasses['DVisualizer'] = DVisualizer

    return exportedClasses

