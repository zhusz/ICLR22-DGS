# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from UDLv3 import udl
from datasets_registration import datasetRetrieveList
from torch.nn.functional import grid_sample
from codes_py.toolbox_show_draw.html_v1 import HTMLStepperNoPrinting
from codes_py.toolbox_3D.pyrender_wrapper_v1 import PyrenderManager
from codes_py.toolbox_3D.view_query_generation_v1 import gen_viewport_query_given_bound
from codes_py.toolbox_3D.representation_v1 import voxSdfSign2mesh_skmc, directDepthMap2PCNP
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP, vertInfo2faceVertInfoTHGPU
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw, getImshow
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly, dumpPly2, dumpPlyPointCloud
from codes_py.toolbox_3D.depth_v1 import depthMap2mesh
from collections import OrderedDict
import cc3d
import os
import copy
import math
import numpy as np

# particular places for the benchmarking code package
from Bpscannet.csvGeometry3DEntry.benchmarkingGeometry3DScannet import benchmarkingGeometry3DScannetFunc


bt = lambda s: s[0].upper() + s[1:]


class Visualizer(object):
    def __init__(self, **kwargs):
        super(Visualizer, self).__init__()
        visualDir = kwargs.get('visualDir', None)
        cudaDevice = kwargs.get('cudaDevice', 'cuda:0')
        ifTracked = kwargs['ifTracked']
        self.visualDir = visualDir
        self.cudaDevice = cudaDevice
        self.ifTracked = ifTracked

        self.htmlStepperMonitor = HTMLStepperNoPrinting(self.visualDir, 50, 'htmlMonitor')

        self.pyrenderManager = PyrenderManager(256, 192)

    def setVisualMeta(self, **kwargs):
        voxSize = kwargs['voxSize']

        # Related to benchmarking. Be careful when modifying them.
        self.numMeshSamplingPoint = 200000
        self.voxSize = voxSize
        self.benchmarkingTightened = 1. - 4. / self.voxSize  # avoid marching cube artifacts
        self.prDistThre = 0.05

        # related to visualization or very marginally affecting benchmarkings (e.g. voxelSize)
        self.verboseGeneralMonitor = 1 if self.ifTracked else 0
        self.verboseBatchForwardingMonitor = 0 if self.ifTracked else 0

        self.verboseGeneralDemo = 1 if self.ifTracked else 0
        self.verboseBatchForwardingDemo = 0

    @staticmethod
    def doQueryPred0(img0, **kwargs):
        queryPointCam0 = kwargs['queryPointCam0']
        queryPointCamPersp0 = kwargs['queryPointCamPersp0']
        models = kwargs['models']
        cudaDevice = kwargs['cudaDevice']
        batchSize = kwargs.get('batchSize', 100 ** 2)
        verboseBatchForwarding = kwargs.get('verboseBatchForwarding')

        assert len(queryPointCam0.shape) == 2 and queryPointCam0.shape[1] == 3
        assert len(queryPointCamPersp0.shape) == 2 and queryPointCamPersp0.shape[1] == 3
        assert len(img0) == 3 and img0.shape[0] == 3

        encoder = models['encoder']
        decoder = models['decoder']
        positionalEncoder = models['positionalEncoder']
        encoder.eval()
        decoder.eval()
        positionalEncoder.eval()
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
        elif useSurfaceOrDepth == 'depth':
            tmp2 = depthMap2mesh(
                tmp1['depthPred'], bsv0['fScaleWidth'], bsv0['fScaleHeight'],
                cutLinkThre=0.05 * 2,
            )
            if tmp2['face0'].shape[0] == 0:
                tmp2['vertCam0'] = np.eye(3).astype(np.float32)
                tmp2['face0'] = np.array([[0, 1, 2]], dtype=np.int32)
            bsv0['predVertCam'] = tmp2['vertCam0']
            bsv0['predFace'] = tmp2['face0']
        elif useSurfaceOrDepth == 'postprocess1':
            predVertCam0, predFace0 = cls.postprocess1(tmp0, tmp1, cudaDevice=cudaDevice)
            bsv0['predVertCam'] = predVertCam0
            bsv0['predFace'] = predFace0
        else:
            raise NotImplementedError('Unknown useSurfaceOrDepth: %s' % useSurfaceOrDepth)
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

    def stepMonitorTrain(self, bsv0, **kwargs):
        # in-place batch benchmarking
        # not for any particular benchmarks
        meta = kwargs['meta']
        models = kwargs['models']
        datasets = kwargs['datasets']
        Trainer = kwargs['Trainer']
        ifRequiresDrawing = kwargs['ifRequiresDrawing']
        ifRequiresPredictingHere = kwargs['ifRequiresPredictingHere']

        verboseBatchForwarding = self.verboseBatchForwardingMonitor

        bsv0_first = benchmarkingGeometry3DScannetFunc(
            copy.deepcopy(bsv0), scannetMeshCache=meta['scannetMeshCache'],
            datasets=datasets,

            # misc
            cudaDevice=self.cudaDevice,
            raiseOrTrace='raise',

            # benchmarking rules parameters
            voxSize=self.voxSize, numMeshSamplingPoint=self.numMeshSamplingPoint,

            # drawing
            ifRequiresDrawing=ifRequiresDrawing,
            pyrenderManager=self.pyrenderManager,

            # predicting
            ifRequiresPredictingHere=ifRequiresPredictingHere,
            models=models, verboseBatchForwarding=verboseBatchForwarding,
            Trainer=Trainer, doPred0Func=self.doPred0, useSurfaceOrDepth='surface',
        )
        # If you wish other type of benchmarking variants, you can append in here.

        bsv0 = bsv0_first

        return bsv0

    def htmlMonitor(self, bsv0, **kwargs):
        visualDir = self.visualDir
        did = int(bsv0['did'])
        index = int(bsv0['index'])
        visIndex = int(bsv0['visIndex'])
        houseID = int(bsv0['houseID'])
        datasetID = int(bsv0['datasetID'])
        flagSplit = int(bsv0['flagSplit'])
        dataset = datasetRetrieveList[datasetID]
        P, D, S, R, iterCount = bsv0['P'], bsv0['D'], bsv0['S'], bsv0['R'], bsv0['iterCount']
        houseID = int(bsv0['houseID'])
        viewID = int(bsv0['viewID'])
        projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'

        assert dataset.startswith('scannetGivenRender')
        cache_dataset_root = projRoot + 'remote_fastdata/cache/scannet/'
        A1_house = udl('pkl_A1_', 'scannet')
        fileList_house = A1_house['fileList']
        scanIDList_house = A1_house['scanIDList']

        summary0 = OrderedDict([])
        txt0 = []
        brInds = [0]

        # load original Image and Depth
        tmp = bsv0['imgForUse'].transpose((1, 2, 0))
        summary0['imgForUse'] = tmp
        txt0.append('Shape: %s' % str(bsv0['imgForUse'].shape))
        tmp = bsv0['depthForUse'][0]
        summary0['depthForUse'] = getImshow(tmp, vmin=0, vmax=3)
        txt0.append('Shape: %s' % str(tmp.shape))
        if 'depthPred' in bsv0.keys():
            tmp = bsv0['depthPred'][0]
            # summary0['depthPred'] = getImshow(tmp, vmin=0, vmax=3)
            summary0['depthPred'] = getImshow(tmp, cmap='inferno')
            txt0.append('Shape: %s' % str(tmp.shape))
            tmp = bsv0['depthPred'][0] - bsv0['depthForUse'][0]
            # summary0['depthDelta'] = getImshow(tmp, vmin=-1, vmax=1)
            summary0['depthDelta'] = getImshow(tmp, cmap='inferno')
            txt0.append('Shape: %s, absDepthEval: %.3f' %
                        (str(tmp.shape), np.abs(tmp[np.isfinite(tmp)]).mean()))

        brInds.append(len(summary0))

        # render the mesh in here, and dump the query overlay in below.
        for meshName in ['evalPred', 'evalView']:
            sysLabel = 'world'
            meshVertInfoFaceInfoType = 'vertRgb'
            prefix = '%s%s%sMeshDrawingPackage' % (meshName, bt(sysLabel), bt(meshVertInfoFaceInfoType))
            numView = int(bsv0[prefix + 'NumView'])

            summary0['%s Floor Plan' % meshName] = bsv0[prefix + 'FloorPlan']
            if ('%sPCBenF1' % meshName) in bsv0.keys():
                txt0.append('P: %.3f, R: %.3f, F1: %.3f' % (
                    bsv0['%sPCBenPrec' % meshName], bsv0['%sPCBenRecall' % meshName],
                    bsv0['%sPCBenF1' % meshName]
                ))
            else:
                txt0.append('')

            for v in range(numView):
                summary0['%s View %d Color' % (meshName, v)] = bsv0[prefix + 'ViewColor'][v, :, :, :]
                txt0.append('Shape: %s' % str(bsv0[prefix + 'ViewColor'][v].shape))
            brInds.append(len(summary0))

        self.htmlStepperMonitor.step2(
            summary0=summary0, txt0=txt0, brInds=brInds,
            headerMessage='Dataset: %s, Index: %s, flagSplit: %d, %s-%s-%s-%s Iter %d, visIndex: %d' %
                          (dataset, index, flagSplit, P, D, S, R, iterCount, visIndex),
            subMessage='HouseID %d, ViewID %d' % (houseID, viewID),
        )

    def dumpMonitor(self, bsv0, **kwargs):
        visualDir = self.visualDir
        did = int(bsv0['did'])
        index = int(bsv0['index'])
        houseID = int(bsv0['houseID'])
        datasetID = int(bsv0['datasetID'])
        flagSplit = int(bsv0['flagSplit'])
        dataset = datasetRetrieveList[datasetID]
        P, D, S, R, iterCount = bsv0['P'], bsv0['D'], bsv0['S'], bsv0['R'], bsv0['iterCount']

        fn_prefix = '%s-%s(%s)-%s%s%s%sI%d' % (dataset, index, flagSplit, P, D, S, R, iterCount)

        for meshName in ['evalView', 'evalPred']:
            sysLabel = 'cam'
            # dumpPly(
            #     visualDir + '%s-%s(%s)-vertRgb.ply' % (fn_prefix, meshName, sysLabel),
            #     bsv0['%sVert%s' % (meshName, bt(sysLabel))],
            #     bsv0['%sFace' % meshName],
            #     bsv0['%sVertRgb' % meshName],
            #     )
            dumpPly2(
                visualDir + '%s-%s(%s)-faceRgb.ply' % (fn_prefix, meshName, sysLabel),
                bsv0['%sVert%s' % (meshName, bt(sysLabel))],
                bsv0['%sFace' % meshName],
                bsv0['%sFaceRgb' % meshName],
            )
