# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP
from codes_py.np_ext.mat_io_v1 import pSaveMat
from plyfile import PlyData, PlyElement
from codes_py.py_ext.misc_v1 import mkdir_full
from codes_py.np_ext.array_mode_v1 import mode
import numpy as np
import csv
import json
from matplotlib.cm import get_cmap
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly2, dumpPly, load_ply_np


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    assert dataset == 'scannet'
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
    scannetRawRoot = projRoot + 'remote_fastdata/scannet/'
    visualDir = projRoot + 'cache/dataset/%s/R17/' % dataset
    os.makedirs(visualDir, exist_ok=True)

    A1 = udl('pkl_A1_', 'scannet')
    m = A1['m']
    fileList = A1['fileList']
    scanIDList = A1['scanIDList']
    cmap = get_cmap('inferno')

    j1 = int(os.environ['J1'])
    j2 = int(os.environ['J2'])
    assert 0 <= j1 < j2

    indChosen = list(range(j1, j2))
    indVisChosen = [0, 1, 2, 3, 4, 7, 10, 20, 30]

    flagSplit = udl('_A01_flagSplit', 'scannet')

    for j in indChosen:
        print('Processing R17 for %s-%d (%s) J1=%d, J2=%d' %
              (dataset, j, scanIDList[j], j1, j2))

        # vert and face (now using ply own utilities)
        if flagSplit[j] < 3:
            vertWorld0A, face0A = load_ply_np(
                scannetRawRoot + '%s%s_vh_clean_2.ply' % (fileList[j], scanIDList[j]))
            vertWorld0B, face0B = load_ply_np(
                scannetRawRoot + '%s%s_vh_clean_2.labels.ply' % (fileList[j], scanIDList[j]))

            assert np.all(vertWorld0A == vertWorld0B)
            assert np.all(face0A == face0B)
            vertWorld0 = vertWorld0A.copy()
            face0 = face0A.copy()
            nVert = vertWorld0.shape[0]
            nFace = face0.shape[0]
            del vertWorld0A, vertWorld0B, face0A, face0B

            # obtaining objID from json
            with open(scannetRawRoot + '%s%s_vh_clean_2.0.010000.segs.json' %
                      (fileList[j], scanIDList[j]), 'r') as f:
                s1 = json.load(f)
            with open(scannetRawRoot + '%s%s.aggregation.json' %
                      (fileList[j], scanIDList[j]), 'r') as f:
                s2 = json.load(f)
            '''  scannet-18 does not satisfy, but we let it go somehow.
            if not s2['segmentsFile'] == ('scannet.%s_vh_clean_2.0.010000.segs.json' %
                                          scanIDList[j]):
                print('Problem occurred. Please check!')
                import ipdb
                ipdb.set_trace()
                print(1 + 1)
            '''
            segIndices = np.array(s1['segIndices'], dtype=np.int32)
            assert segIndices.shape[0] == nVert
            vertObjID0 = -np.ones((nVert,), dtype=np.int32)
            for g, sg in enumerate(s2['segGroups']):
                vertObjID0[
                    np.isin(segIndices, np.array(sg['segments'], dtype=np.int32))] = sg['objectId']
                assert sg['objectId'] == sg['id']

            ply = PlyData.read(scannetRawRoot + '%s%s_vh_clean_2.labels.ply' %
                               (fileList[j], scanIDList[j]))
            vertNyu40ID0 = ply['vertex']['label']
            assert vertNyu40ID0.shape[0] == nVert

            # vertInfo to faceInfo
            faceVertNyu40ID0 = vertInfo2faceVertInfoNP(
                vertNyu40ID0[None, :, None].astype(np.float32), face0[None])\
                [0, :, 0].astype(np.int32)
            nyu40ID0 = np.apply_along_axis(mode, 1, faceVertNyu40ID0)
            faceVertObjID0 = vertInfo2faceVertInfoNP(
                vertObjID0[None, :, None].astype(np.float32), face0[None])\
                [0, :, 0].astype(np.int32)
            objID0 = np.apply_along_axis(mode, 1, faceVertObjID0)

            # other detailed mesh components
            vert0 = vertWorld0
            faceVert0 = vertInfo2faceVertInfoNP(
                vert0[None], face0[None],
            )[0]
            faceCentroid0 = faceVert0.mean(1)
            faceNormal0 = np.cross(
                faceVert0[:, 1, :] - faceVert0[:, 0, :],
                faceVert0[:, 2, :] - faceVert0[:, 0, :],
            )
            faceNormal0 = np.divide(
                faceNormal0,
                np.linalg.norm(faceNormal0, ord=2, axis=1)[:, None] + 1.e-6,
            )
            pSaveMat(projRoot + 'v/R/%s/R17/%08d.mat' % (dataset, j), {
                'vert0': vert0,
                'face0': face0,
                'faceCentroid0': faceCentroid0,
                'faceNormal0': faceNormal0,
                'faceObjID0': objID0,
                'faceNyu40ID0': nyu40ID0,
            })
        else:
            vert0, face0 = load_ply_np(
                scannetRawRoot + '%s%s_vh_clean_2.ply' % (fileList[j], scanIDList[j])
            )
            nVert = vert0.shape[0]
            nFace = face0.shape[0]

            # other detailed mesh components
            faceVert0 = vertInfo2faceVertInfoNP(
                vert0[None], face0[None],
            )[0]
            faceCentroid0 = faceVert0.mean(1)
            faceNormal0 = np.cross(
                faceVert0[:, 1, :] - faceVert0[:, 0, :],
                faceVert0[:, 2, :] - faceVert0[:, 0, :],
            )
            faceNormal0 = np.divide(
                faceNormal0,
                np.linalg.norm(faceNormal0, ord=2, axis=1)[:, None] + 1.e-6,
            )
            pSaveMat(projRoot + 'v/R/%s/R17/%08d.mat' % (dataset, j), {
                'vert0': vert0,
                'face0': face0,
                'faceCentroid0': faceCentroid0,
                'faceNormal0': faceNormal0,
            })

        if j in indVisChosen:
            assert flagSplit[j] < 3
            # dump ObjID0
            vertObjIDColor0 = cmap(vertObjID0.astype(np.float32) / vertObjID0.max())[:, :3]
            vertObjIDColor0[vertObjID0 == -1, :] = 0
            dumpPly(visualDir + 'vertObjID_%08d.ply' % j, vertWorld0, face0, vertObjIDColor0)
            objIDColor0 = cmap(objID0.astype(np.float32) / vertObjID0.max())[:, :3]
            objIDColor0[objID0 == -1, :] = 0
            dumpPly2(visualDir + 'objID_%08d.ply' % j, vertWorld0, face0, objIDColor0)

            # dump nyu40
            assert vertNyu40ID0.min() >= 0 and vertNyu40ID0.max() <= 40
            vertNyu40IDColor0 = cmap(vertNyu40ID0.astype(np.float32) / 40)[:, :3]
            vertNyu40IDColor0[vertNyu40ID0 == 0, :] = 0
            dumpPly(visualDir + 'vertNyu40ID_%08d.ply' % j, vertWorld0, face0, vertNyu40IDColor0)
            nyu40IDColor0 = cmap(nyu40ID0.astype(np.float32) / 40)[:, :3]
            nyu40IDColor0[nyu40ID0 == 0, :] = 0
            dumpPly2(visualDir + 'nyu40ID_%08d.ply' % j, vertWorld0, face0, nyu40IDColor0)


if __name__ == '__main__':
    main()
