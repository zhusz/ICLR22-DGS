# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.functional import grid_sample
from UDLv3 import udl
from codes_py.template_visualStatic.visualStatic_v1 import TemplateVisualStatic
from codes_py.toolbox_show_draw.html_v1 import HTMLStepperNoPrinting
from codes_py.toolbox_3D.rotations_v1 import ELU02cam0, camSys2CamPerspSys0
from codes_py.toolbox_show_draw.draw_v1 import getPltDraw
from codes_py.toolbox_show_draw.show3D_v1 import showPoint3D4
from codes_py.toolbox_3D.rotations_v1 import getRotationMatrixBatchNP
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP
from collections import OrderedDict
from easydict import EasyDict
import numpy as np
import math
import copy
import pickle
from datasets_registration import datasetDict
np.set_printoptions(suppress=True)


from . import resnet50


bt = lambda s: s[0].upper() + s[1:]


class Visualizer(TemplateVisualStatic):
    def __init__(self, **kwargs):
        super(Visualizer, self).__init__()
        visualDir = kwargs.get('visualDir', None)
        cudaDevice = kwargs.get('cudaDevice', 'cuda:0')
        ifTracked = kwargs['ifTracked']
        self.visualDir = visualDir
        self.cudaDevice = cudaDevice
        self.ifTracked = ifTracked

        self.htmlStepperDataset = HTMLStepperNoPrinting(self.visualDir, 50, 'htmlDataset')
        self.htmlStepperMonitor = HTMLStepperNoPrinting(self.visualDir, 50, 'htmlMonitor')
        self.htmlStepperHolistic = HTMLStepperNoPrinting(self.visualDir, 50, 'htmlHolistic')
        self.htmlStepperDetached = HTMLStepperNoPrinting(self.visualDir, 50, 'htmlDetached')

    def setVisualMeta(self):
        self.verboseGeneralMonitor = 1 if self.ifTracked else 0
        self.verboseBatchForwardingMonitor = 0 if self.ifTracked else 0
        self.camcubeVoxSizeMonitor = 64
        self.verboseGeneralHolistic = 1 if self.ifTracked else 0
        self.verboseGeneralBp12 = 0 if self.ifTracked else 0
        self.verboseBatchForwardingBp12 = 0 if self.ifTracked else 0
        self.camcubeVoxSizeBp12 = 64
        self.numPointToPlotBp12 = 50000

    @staticmethod
    def doQueryPred0(img0, queryPointCam0, queryPointCamPersp0,
                     **kwargs):

        meta = kwargs['meta']
        models = kwargs['models']
        cudaDevice = kwargs['cudaDevice']
        batchSize = kwargs.get('batchSize', 100 ** 2)
        verboseBatchForwarding = kwargs.get('verboseBatchForwarding', 1)
        samplingFromNetwork1ToNetwork2Func = kwargs['samplingFromNetwork1ToNetwork2Func']
        samplingFromNetwork2ToNetwork3Func = kwargs['samplingFromNetwork2ToNetwork3Func']

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

        predBisem0 = np.zeros((nQuery, ), dtype=np.float32)
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

    def stepPredForwardCamcubeVersionCorenet(self, bsv0, **kwargs):
        gridStandardCamcubeName = kwargs['gridStandardCamcubeName']
        outputCamcubeName = kwargs['outputCamcubeName']
        outputMeshName = kwargs['outputMeshName']
        verboseBatchForwarding = kwargs['verboseBatchForwarding']
        verboseGeneral = kwargs['verboseGeneral']
        samplingFromNetwork1ToNetwork2Func = kwargs['samplingFromNetwork1ToNetwork2Func']
        samplingFromNetwork2ToNetwork3Func = kwargs['samplingFromNetwork2ToNetwork3Func']
        meta = kwargs['meta']
        cudaDevice = kwargs['cudaDevice']
        models = kwargs['models']
        if kwargs['verboseGeneral'] > 0:
            print('[Visualizer] stepPredForwardCamcubeVersionCorenet'
                  '(outputCamcubeName = %s, gridStandardCamcubeName = %s' %
                  (outputCamcubeName, gridStandardCamcubeName))

        for k in ['VoxXyzCam', 'VoxXyzCamPersp', 'GoxyzCam', 'SCell', 'Maskfloat']:
            bsv0[outputCamcubeName + 'Camcube' + k] = \
                copy.deepcopy(bsv0[gridStandardCamcubeName + 'Camcube' + k])

        fullQueryVoxXyzCam0 = bsv0[outputCamcubeName + 'CamcubeVoxXyzCam']
        Ly, Lx, Lz, _ = fullQueryVoxXyzCam0.shape
        fullQueryVoxXyzCam0 = fullQueryVoxXyzCam0.reshape((-1, 3))
        fullQueryVoxXyzCamPersp0 = bsv0[outputCamcubeName + 'CamcubeVoxXyzCamPersp'].reshape((-1, 3))
        fullQueryMaskfloat0 = bsv0[outputCamcubeName + 'CamcubeMaskfloat'].reshape((-1))

        tmp = self.doQueryPred0(
            bsv0['img'], fullQueryVoxXyzCam0, fullQueryVoxXyzCamPersp0,
            meta=meta, models=models, cudaDevice=cudaDevice,
            verboseBatchForwarding=verboseBatchForwarding,
            samplingFromNetwork1ToNetwork2Func=samplingFromNetwork1ToNetwork2Func,
            samplingFromNetwork2ToNetwork3Func=samplingFromNetwork2ToNetwork3Func,
        )
        fullQueryOccfloat0 = tmp['occfloat']
        fullQueryOccfloat0[fullQueryMaskfloat0 <= 0] = 1.  # empty
        fullQueryBisem0 = tmp['bisem']
        fullQueryBisem0[fullQueryMaskfloat0 <= 0] = 0.  # empty

        bsv0[outputCamcubeName + 'CamcubePredOccfloatYXZ'] = fullQueryOccfloat0.reshape((Ly, Lx, Lz))
        bsv0[outputCamcubeName + 'CamcubePredBisemYXZ'] = fullQueryBisem0.reshape((Ly, Lx, Lz))

        bsv0 = self.stepCubeOccfloatToPlainMesh(
            bsv0, cubeType='camcube', inputCubeName=outputCamcubeName, outputMeshName=outputMeshName,
            occfloatKey='predOccfloatYXZ', verboseGeneral=verboseGeneral,
        )
        return bsv0

    def stepMonitor(self, bsv0, **kwargs):
        meta = kwargs['meta']
        models = kwargs['models']
        datasets = kwargs['datasets']
        generalVoxelizationFunc = kwargs['generalVoxelizationFunc']
        samplingFromNetwork1ToNetwork2Func = kwargs['samplingFromNetwork1ToNetwork2Func']
        samplingFromNetwork2ToNetwork3Func = kwargs['samplingFromNetwork2ToNetwork3Func']

        # from bsv0
        index = int(bsv0['index'])
        did = int(bsv0['did'])

        # from meta
        f0 = meta['f0']

        # from models
        encoder = models['encoder']
        networkTwo = models['networkTwo']
        encoder.eval()
        networkTwo.eval()

        # from self
        camcubeVoxSize = self.camcubeVoxSizeMonitor
        verboseGeneral = self.verboseGeneralMonitor
        verboseBatchForwarding = self.verboseBatchForwardingMonitor
        cudaDevice = self.cudaDevice

        # standard
        bsv0 = self.stepCamcubeGridStandardVersionCorenet(
            bsv0, outputCamcubeName='corenet', camcubeVoxSize=camcubeVoxSize,
            f0=f0, verboseGeneral=verboseGeneral,
        )

        # load mesh
        tmp1 = datasets[did].getRawMeshCamOurs({}, index)
        bsv0['rawVertCam'] = tmp1['vertCamOurs'].astype(np.float32)
        bsv0['rawFace'] = tmp1['face'].astype(np.int32)
        bsv0['rawVertRgb'] = np.ones_like(bsv0['rawVertCam']) * 0.6

        # label
        bsv0['corenetCamcubeLabelBisemZYX'] = generalVoxelizationFunc(
            [vertInfo2faceVertInfoNP(
                np.stack([
                    bsv0['rawVertCam'][:, 0] + 0.5,
                    bsv0['rawVertCam'][:, 1] + 0.5,
                    bsv0['rawVertCam'][:, 2] - f0 / 2.,
                ], 1)[None, :, :],
                bsv0['rawFace'][None, :, :],
            )[0]],
            camcubeVoxSize,
            camcubeVoxSize,
            camcubeVoxSize,
            cudaDevice,
        )[0].detach().cpu().numpy()
        bsv0['corenetCamcubeLabelOccfloatZYX'] = 1. - bsv0['corenetCamcubeLabelBisemZYX']
        bsv0['corenetCamcubeLabelOccfloatYXZ'] = bsv0['corenetCamcubeLabelOccfloatZYX'].transpose(
            (1, 2, 0))
        bsv0['corenetCamcubeLabelBisemYXZ'] = bsv0['corenetCamcubeLabelBisemZYX'].transpose(
            (1, 2, 0))
        bsv0 = self.stepCubeOccfloatToPlainMesh(
            bsv0, cubeType='camcube', inputCubeName='corenet',
            outputMeshName='corenetLabel',
            occfloatKey='labelOccfloatYXZ', verboseGeneral=self.verboseGeneralMonitor,
        )

        # pred
        bsv0 = self.stepPredForwardCamcubeVersionCorenet(
            bsv0, gridStandardCamcubeName='corenet', outputCamcubeName='corenet',
            outputMeshName='corenetPred',
            verboseGeneral=verboseGeneral,
            verboseBatchForwarding=verboseBatchForwarding,
            samplingFromNetwork1ToNetwork2Func=samplingFromNetwork1ToNetwork2Func,
            samplingFromNetwork2ToNetwork3Func=samplingFromNetwork2ToNetwork3Func,
            meta=meta,
            cudaDevice=cudaDevice, models=models,
        )

        # benchmarking computation
        bsv0 = self.stepCubePairToIou(
            bsv0, inputPredCubeName='corenet', inputLabelCubeName='corenet', cubeType='camcube',
            outputBenName='corenetCubeIou', predJudgeTag='predOccfloatYXZ',
            labelJudgeTag='labelOccfloatYXZ',
            judgeFunc=lambda x: (x < 0.5).astype(bool),
            verboseGeneral=verboseGeneral,
        )

        bsv0 = self._stepDspecificMonitor(bsv0)

        # mesh to drawing
        bsv0['ECam'] = np.array([0, 0, 0], dtype=np.float32)
        bsv0['LCam'] = np.array([0, 0, 1], dtype=np.float32)
        bsv0['UCam'] = np.array([0, -1, 0], dtype=np.float32)
        bsv0['amodalDepthMax'] = 4.
        bsv0['winHeight'] = float(256)
        bsv0['winWidth'] = float(256)
        bsv0['focalLengthHeight'] = float(128 * f0)
        bsv0['focalLengthWidth'] = float(128 * f0)
        bsv0 = self.stepDrawMeshPackage(
            bsv0, inputMeshName='raw', outputMeshDrawingName='raw2',
            usingObsFromMeshDrawingName=None, numView=3, meshVertInfoFaceInfoType='plain2',
            cudaDevice=self.cudaDevice, sysLabel='cam', verboseGeneral=verboseGeneral,
        )
        bsv0 = self.stepDrawMeshPackage(
            bsv0, inputMeshName='corenetLabel', outputMeshDrawingName='corenetLabel2',
            usingObsFromMeshDrawingName=None, numView=3, meshVertInfoFaceInfoType='plain2',
            cudaDevice=self.cudaDevice, sysLabel='cam', verboseGeneral=verboseGeneral,
        )
        bsv0 = self.stepDrawMeshPackage(
            bsv0, inputMeshName='corenetPred', outputMeshDrawingName='corenetPred2',
            usingObsFromMeshDrawingName='corenetLabel2', numView=3, meshVertInfoFaceInfoType='plain2',
            cudaDevice=self.cudaDevice, sysLabel='cam', verboseGeneral=verboseGeneral,
        )

        return bsv0

    def _stepDspecificMonitor(self, bsv0):
        return bsv0

    def htmlMonitor(self, bsv0, **kwargs):
        if self.ifTracked:
            summary = OrderedDict([])
            txt0 = []
            brInds = [0]

            summary['Input RGB'] = bsv0['img'].transpose((1, 2, 0))[None, :, :, :]
            txt0.append('flagSplit: %d' % bsv0['flagSplit'])
            brInds.append(len(summary))

            Lz = bsv0['corenetCamcubePredBisemYXZ'].shape[2]
            for depthID in [int(Lz * 0.375), int(Lz * 0.5), int(Lz * 0.625)]:
                summary['Pred BiSem Depth %d' % depthID] = getPltDraw(
                    lambda ax: ax.imshow(bsv0['corenetCamcubePredBisemYXZ'].transpose(
                        (2, 0, 1))[depthID, :, :]))[None]
                txt0.append('')
            brInds.append(len(summary))
            for depthID in [int(Lz * 0.375), int(Lz * 0.5), int(Lz * 0.625)]:
                summary['Label BiSem Depth %d' % depthID] = getPltDraw(
                    lambda ax: ax.imshow(bsv0['corenetCamcubeLabelBisemYXZ'].transpose(
                        (2, 0, 1))[depthID, :, :]))[None]
                txt0.append('')
            brInds.append(len(summary))

            for v in range(bsv0['corenetPred2CamPlain2Drawing0'].shape[0]):
                summary['Pred View %d' % v] = bsv0['corenetPred2CamPlain2Drawing0'][v, :, :, :][None]
                txt0.append('')
            brInds.append(len(summary))

            for v in range(bsv0['corenetLabel2CamPlain2Drawing0'].shape[0]):
                summary['Label View %d' % v] = bsv0['corenetLabel2CamPlain2Drawing0'][v, :, :, :][None]
                txt0.append('')
            brInds.append(len(summary))

            visual = EasyDict()
            visual.summary = summary
            visual.m = 1
            visual.datasetID = [bsv0['datasetID']]
            visual.dataset = [bsv0['dataset']]
            visual.index = [bsv0['index']]
            visual.insideBatchIndex = [bsv0['visIndex']]
            self.htmlStepperMonitor.step(bsv0['iterCount'], None, visual, txts=[txt0], brInds=brInds)

    def dumpMonitor(self, bsv0, **kwargs):
        verboseGeneral = self.verboseGeneralMonitor
        if self.ifTracked:
            iterCount = bsv0['iterCount']
            if iterCount < 50:
                self.dumpMeshAsPly(
                    bsv0, visualDir=self.visualDir, meshName='raw',
                    meshVertInfoFaceInfoType='plain', sysLabel='cam',
                    verboseGeneral=verboseGeneral,
                )

    def stepOneBp12(self, bsv0, **kwargs):
        meta = kwargs['meta']
        models = kwargs['models']
        datasets = kwargs['datasets']
        generalVoxelizationFunc = kwargs['generalVoxelizationFunc']
        samplingFromNetwork1ToNetwork2Func = kwargs['samplingFromNetwork1ToNetwork2Func']
        samplingFromNetwork2ToNetwork3Func = kwargs['samplingFromNetwork2ToNetwork3Func']

        # from bsv0
        index = int(bsv0['index'])
        did = int(bsv0['did'])

        # from meta
        f0 = meta['f0']

        # from models
        encoder = models['encoder']
        networkTwo = models['networkTwo']
        encoder.eval()
        networkTwo.eval()

        # from self
        camcubeVoxSize = self.camcubeVoxSizeBp12
        verboseGeneral = self.verboseGeneralBp12
        verboseBatchForwarding = self.verboseBatchForwardingBp12
        cudaDevice = self.cudaDevice

        # standard
        bsv0 = self.stepCamcubeGridStandardVersionCorenet(
            bsv0, outputCamcubeName='corenet', camcubeVoxSize=camcubeVoxSize,
            f0=f0, verboseGeneral=verboseGeneral,
        )

        tmp1 = datasets[did].getRawMeshCamOurs({}, index)
        bsv0['rawVertCam'] = tmp1['vertCamOurs'].astype(np.float32)
        bsv0['rawFace'] = tmp1['face'].astype(np.int32)
        bsv0['rawVertRgb'] = np.ones_like(bsv0['rawVertCam']) * 0.6

        # label
        bsv0['corenetCamcubeLabelBisemZYX'] = generalVoxelizationFunc(
            [vertInfo2faceVertInfoNP(
                np.stack([
                    bsv0['rawVertCam'][:, 0] + 0.5,
                    bsv0['rawVertCam'][:, 1] + 0.5,
                    bsv0['rawVertCam'][:, 2] - f0 / 2.,
                ], 1)[None, :, :],
                bsv0['rawFace'][None, :, :],
            )[0]],
            camcubeVoxSize,
            camcubeVoxSize,
            camcubeVoxSize,
            cudaDevice,
        )[0].detach().cpu().numpy()
        bsv0['corenetCamcubeLabelOccfloatZYX'] = 1. - bsv0['corenetCamcubeLabelBisemZYX']
        bsv0['corenetCamcubeLabelOccfloatYXZ'] = bsv0['corenetCamcubeLabelOccfloatZYX'].transpose(
            (1, 2, 0))
        bsv0['corenetCamcubeLabelBisemYXZ'] = bsv0['corenetCamcubeLabelBisemZYX'].transpose(
            (1, 2, 0))
        bsv0 = self.stepCubeOccfloatToPlainMesh(
            bsv0, cubeType='camcube', inputCubeName='corenet',
            outputMeshName='corenetLabel',
            occfloatKey='labelOccfloatYXZ', verboseGeneral=self.verboseGeneralMonitor,
        )

        # pred
        bsv0 = self.stepPredForwardCamcubeVersionCorenet(
            bsv0, gridStandardCamcubeName='corenet', outputCamcubeName='corenet',
            outputMeshName='corenetPred',
            verboseGeneral=verboseGeneral,
            verboseBatchForwarding=verboseBatchForwarding,
            samplingFromNetwork1ToNetwork2Func=samplingFromNetwork1ToNetwork2Func,
            samplingFromNetwork2ToNetwork3Func=samplingFromNetwork2ToNetwork3Func,
            meta=meta,
            cudaDevice=cudaDevice, models=models,
        )

        # benchmarking computation
        bsv0 = self.stepCubePairToIou(
            bsv0, inputPredCubeName='corenet', inputLabelCubeName='corenet', cubeType='camcube',
            outputBenName='corenetCubeIou', predJudgeTag='predOccfloatYXZ',
            labelJudgeTag='labelOccfloatYXZ',
            judgeFunc=lambda x: (x < 0.5).astype(bool),
            verboseGeneral=verboseGeneral,
        )

        # packbit
        corenetCamcubePredBisemYXZBool0 = (bsv0['corenetCamcubePredBisemYXZ'] > 0.5).astype(bool)
        corenetCamcubePredBisemYXZPacked0 = np.packbits(
            corenetCamcubePredBisemYXZBool0.reshape((-1)), axis=None, bitorder='big')
        bsv0['corenetCamcubePredBisemYXZPacked'] = corenetCamcubePredBisemYXZPacked0

        bsv0['corenetCamcubeLabelBisemYXZPacked'] = np.packbits(
            bsv0['corenetCamcubeLabelBisemYXZ'].reshape((-1)).astype(bool), axis=None, bitorder='big'
        )
        return bsv0

    def stepBenBp12(self, bsv0, **kwargs):
        datasets = kwargs['datasets']
        did = int(bsv0['did'])
        index = int(bsv0['index'])
        bsv0['catName'] = datasets[did].catNameList[index]
        bsv0['catID'] = datasets[did].catIDList[index]

        bsv0['iou'] = bsv0['corenetCubeIou']
        return bsv0

    def stepHDcBp12_index_dict(self, **kwargs):
        datasetObj = kwargs['datasetObj']
        index = kwargs['index']
        corenetCamcubeLabelBisemYXZPacked = kwargs['corenetCamcubeLabelBisemYXZPacked']
        camcubeVoxSizeBp12 = self.camcubeVoxSizeBp12

        batch0_np = datasetObj.getOneNP(index)
        bsv0 = batch0_np

        f0 = udl('pkl_A1b_f0', 'corenetChoySingleRendering')
        bsv0['ECam'] = np.array([0, 0, 0], dtype=np.float32)
        bsv0['LCam'] = np.array([0, 0, 1], dtype=np.float32)
        bsv0['UCam'] = np.array([0, -1, 0], dtype=np.float32)
        bsv0['EWorld'] = np.array([0, 0, 0], dtype=np.float32)
        bsv0['LWorld'] = np.array([0, 0, 1], dtype=np.float32)
        bsv0['UWorld'] = np.array([0, -1, 0], dtype=np.float32)
        bsv0['amodalDepthMax'] = 4.
        bsv0['winHeight'] = float(256)
        bsv0['winWidth'] = float(256)
        bsv0['focalLengthHeight'] = float(128 * f0)
        bsv0['focalLengthWidth'] = float(128 * f0)

        bsv0['corenetCamcubeLabelBisemYXZ'] = np.unpackbits(
            corenetCamcubeLabelBisemYXZPacked, axis=None, count=None, bitorder='big',
        ).reshape((camcubeVoxSizeBp12, camcubeVoxSizeBp12, camcubeVoxSizeBp12))
        bsv0['corenetCamcubeLabelOccfloatYXZ'] = \
            1. - bsv0['corenetCamcubeLabelBisemYXZ']
        bsv0['corenetCamcubeMaskfloat'] = np.ones((camcubeVoxSizeBp12, camcubeVoxSizeBp12, camcubeVoxSizeBp12), dtype=np.float32)
        bsv0['corenetCamcubeGoxyzCam'] = np.array([
            -0.5 + 0.5 / camcubeVoxSizeBp12, -0.5 + 0.5 / camcubeVoxSizeBp12, f0 / 2. + 0.5 / camcubeVoxSizeBp12,
        ], dtype=np.float32)
        bsv0['corenetCamcubeSCell'] = np.array([1. / camcubeVoxSizeBp12] * 3, dtype=np.float32)
        bsv0 = self.stepCubeOccfloatToPlainMesh(
            bsv0, cubeType='camcube', inputCubeName='corenet', outputMeshName='corenetLabel',
            occfloatKey='labelOccfloatYXZ', verboseGeneral=self.verboseGeneralBp12,
        )
        bsv0['corenetLabelVertRgb'] = np.ones_like(bsv0['corenetLabelVertCam'])
        bsv0['corenetLabelVertRgb'][:, :2] = 0.6  # more blue

        # special processing for corenet
        index = int(bsv0['index'])
        camRObj2CamOurs = datasetObj.camRObj2CamOurs[index]
        camTObj2CamOurs = datasetObj.camTObj2CamOurs[index]
        UCam0 = np.dot(
            camRObj2CamOurs, np.array([0, 1, 0], dtype=np.float32))
        UCam0 = UCam0 / np.linalg.norm(UCam0, ord=2)
        LCenterCam0 = np.array([0, 0, camTObj2CamOurs[2]], dtype=np.float32)

        E0 = np.array([0., 0., 0.], dtype=np.float32)
        rot012 = getRotationMatrixBatchNP(
            np.tile(UCam0, (3, 1)), np.array([0, 120, 240], dtype=np.float32),
            cudaDevice=None
        )
        E012 = np.matmul(
            rot012, -LCenterCam0[None, :, None] + np.tile(E0[None, :, None], (3, 1, 1)),
        )[:, :, 0] + LCenterCam0[None, :]
        E3 = E0 - 0.5 * UCam0
        E345 = np.matmul(
            rot012, -LCenterCam0[None, :, None] + np.tile(E3[None, :, None], (3, 1, 1)),
        )[:, :, 0] + LCenterCam0[None, :]
        bsv0['standardCamVertRgbEPick0List'] = np.concatenate([E012, E345], 0)
        bsv0['standardCamVertRgbLCenter0'] = LCenterCam0
        bsv0['standardCamVertRgbNumView'] = 3
        bsv0['UCam'] = UCam0

        bsv0 = self.stepDrawMeshPackage(
            bsv0, inputMeshName='corenetLabel', outputMeshDrawingName='corenetLabel',
            usingObsFromMeshDrawingName='standard', numView=3, meshVertInfoFaceInfoType='vertRgb',
            cudaDevice=self.cudaDevice, sysLabel='cam', verboseGeneral=self.verboseGeneralBp12,
        )
        numPointToPlot = self.numPointToPlotBp12
        bsv0 = self.stepPlainMeshToPointCloud(
            bsv0, inputMeshName='corenetLabel', outputPointCloudName='corenetLabel',
            numPoint=numPointToPlot, sysLabel='cam', cudaDevice=self.cudaDevice,
            verboseGeneral=self.verboseGeneralBp12,
        )

        return bsv0

    def stepHDcBp12(self, bsv0, **kwargs):
        camcubeVoxSizeBp12 = self.camcubeVoxSizeBp12

        bsv0['corenetCamcubePredBisemYXZ'] = np.unpackbits(
            bsv0['corenetCamcubePredBisemYXZPacked'], axis=None, count=None, bitorder='big',
        ).reshape((camcubeVoxSizeBp12, camcubeVoxSizeBp12, camcubeVoxSizeBp12))
        bsv0['corenetCamcubePredOccfloatYXZ'] = \
            1. - bsv0['corenetCamcubePredBisemYXZ']
        bsv0 = self.stepCubeOccfloatToPlainMesh(
            bsv0, cubeType='camcube', inputCubeName='corenet', outputMeshName='corenetPred',
            occfloatKey='predOccfloatYXZ', verboseGeneral=self.verboseGeneralBp12,
        )
        bsv0['corenetPredVertRgb'] = np.ones_like(bsv0['corenetPredVertCam'])
        bsv0['corenetPredVertRgb'][:, 1:] = 0.6  # more red
        bsv0 = self.stepDrawMeshPackage(
            bsv0, inputMeshName='corenetPred', outputMeshDrawingName='corenetPred',
            usingObsFromMeshDrawingName='corenetLabel', numView=3, meshVertInfoFaceInfoType='vertRgb',
            cudaDevice=self.cudaDevice, sysLabel='cam', verboseGeneral=self.verboseGeneralBp12,
        )
        bsv0 = self.stepPlainMeshToPointCloud(
            bsv0, inputMeshName='corenetPred', outputPointCloudName='corenetPred',
            numPoint=self.numPointToPlotBp12, sysLabel='cam', cudaDevice=self.cudaDevice,
            verboseGeneral=self.verboseGeneralBp12,
        )

        return bsv0

