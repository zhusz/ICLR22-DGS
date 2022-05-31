# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
import numpy as np
from codes_py.toolbox_3D.rotations_v1 import getRotationMatrixBatchTH
from codes_py.toolbox_3D.mesh_io_v1 import load_obj_np, dumpPly, dumpPly2
from codes_py.toolbox_3D.mesh_surgery_v1 import combineMultiShapes_withWhatever
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP
from codes_py.py_ext.misc_v1 import mkdir_full
from codes_py.np_ext.mat_io_v1 import pSaveMat
import pickle
import xml.etree.ElementTree as ET
from codes_py.corenet.geometry.voxelization import voxelize_mesh
from codes_py.corenet.cc import fill_voxels
from matplotlib.cm import get_cmap


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
    rawRoot = projRoot + 'remote_fastdata/openrooms/'
    visualDir = projRoot + 'cache/dataset/%s/dump_A11_combiningMesh/' % dataset
    mkdir_full(visualDir)

    m = udl('_A1_m', dataset)
    tmp = udl('pkl_A1_', dataset)
    sceneList = tmp['sceneList']
    xmlTagList = tmp['xmlTagList']
    del tmp

    semanticOpenrooms = udl('_Z_semanticOpenrooms')
    openroomsSha2name = semanticOpenrooms['openroomsSha2name']
    openrooms60Dict = semanticOpenrooms['openrooms60Dict']

    indChosenDump = list(range(5))
    for j in range(m):
        if j in [502, 522, 952, 1084, 1178, 1479, 1790, 2240]:
            continue
        print('Processing R11a_combiningMesh for %s: %d / %d' % (dataset, j, m))

        fn = rawRoot + 'scenes/%s/%s/main.xml' % (xmlTagList[j], sceneList[j])
        tree = ET.parse(fn)
        scene = tree.getroot()
        assert scene.tag == 'scene'

        vfso0List = []
        objID = 0
        for component in scene:
            if component.tag == 'shape':
                assert component.attrib['type'] == 'obj'

                # filename
                countString = 0
                for child in component:
                    if child.tag == 'string' and child.attrib['name'] == 'filename':
                        objFileName = child.attrib['value'].strip().rstrip()
                        countString += 1
                assert countString == 1
                s = objFileName.split('/')
                for z in range(5):
                    assert s[z] == '..'
                dirTag = s[5]  # layoutMesh or uv_mapped
                if dirTag == 'layoutMesh':
                    sceneTag = s[6]
                    objPureFileName = s[7]
                    if not sceneTag == sceneList[j] or not objPureFileName.endswith('.obj'):
                        print('[Error] discripancy on sceneTag! Please check!')
                        import ipdb
                        ipdb.set_trace()
                    vertObj0, face0 = load_obj_np(rawRoot + '%s/%s/%s' % (
                        dirTag, sceneTag, objPureFileName))
                elif dirTag == 'uv_mapped':
                    catID = s[6]
                    shaID = s[7]
                    objPureFileName = s[8]
                    if not objPureFileName.endswith('.obj'):
                        print('[Error] discripancy on objPureFileName! Please check!')
                        import ipdb
                        ipdb.set_trace()
                    vertObj0, face0 = load_obj_np(rawRoot + '%s/%s/%s/%s' % (
                        dirTag, catID, shaID, objPureFileName))
                else:
                    raise NotImplementedError('Unknown dirTag %s' % dirTag)
                vert0 = vertObj0

                # transform
                countTransform = 0
                for child in component:
                    if child.tag == 'transform':
                        assert child.attrib['name'] == 'toWorld'
                        node = child
                        countTransform += 1
                assert countTransform == 1
                for child in node:
                    if child.tag == 'scale':
                        x = float(child.attrib['x'])
                        y = float(child.attrib['y'])
                        z = float(child.attrib['z'])
                        vert0[:, 0] *= x
                        vert0[:, 1] *= y
                        vert0[:, 2] *= z
                        del x, y, z
                    elif child.tag == 'rotate':
                        angleDegree = child.attrib['angle']
                        x = float(child.attrib['x'])
                        y = float(child.attrib['y'])
                        z = float(child.attrib['z'])
                        norm = (x * x + y * y + z * z) ** 0.5
                        t = max(norm, float(1.e-6))
                        x = x / t
                        y = y / t
                        z = z / t
                        rotationAxis = torch.from_numpy(np.array([[x, y, z]], dtype=np.float32))
                        degrees = torch.from_numpy(np.array([angleDegree], dtype=np.float32))
                        if not np.all(np.isfinite(rotationAxis.detach().cpu().numpy())):
                            print('Wrong')
                            import ipdb
                            ipdb.set_trace()
                            print(1 + 1)
                        rotMat0 = getRotationMatrixBatchTH(
                            rotationAxis, degrees)[0].detach().cpu().numpy()
                        vert0 = np.matmul(vert0, rotMat0.transpose())
                        del angleDegree, x, y, z, norm, t, rotationAxis, degrees, rotMat0
                    elif child.tag == 'translate':
                        x = float(child.attrib['x'])
                        y = float(child.attrib['y'])
                        z = float(child.attrib['z'])
                        vert0[:, 0] += x
                        vert0[:, 1] += y
                        vert0[:, 2] += z
                        del x, y, z
                    else:
                        print('[Error] Unknown type of transform: %s' % 'child.tag')
                        import ipdb
                        ipdb.set_trace()
                        raise NotImplementedError('Unknown type of transform: %s' % child.tag)

                # semantics
                nFace = int(face0.shape[0])
                if dirTag == 'layoutMesh':
                    faceOpenrooms60 = 60 * np.ones((nFace, ), dtype=np.int32)
                elif dirTag == 'uv_mapped':
                    assert catID in openroomsSha2name.keys(), catID
                    faceOpenrooms60 = openrooms60Dict[openroomsSha2name[catID]] * \
                        np.ones((nFace, ), dtype=np.float32)
                else:
                    raise NotImplementedError('Unknown dirTag: %s' % dirTag)
                faceObjID0 = objID * np.ones((nFace, ), dtype=np.int32)
                objID += 1

                vfso0List.append((vert0, face0, faceOpenrooms60, faceObjID0))  # s means semantics

        del vert0, face0, faceOpenrooms60

        # vert/face equivalent
        vertWorld0, face0, faceOpenrooms60o, faceObjID0 = combineMultiShapes_withWhatever(vfso0List)
        faceVertWorld0 = vertInfo2faceVertInfoNP(vertWorld0[None], face0[None])[0]
        # faceCentroid0 = faceVertWorld0.mean(1)
        # Note you cannot use faceCentroid0 for openrooms, as this is big triangles
        tmp0 = np.cross(
            faceVertWorld0[:, 1, :] - faceVertWorld0[:, 0, :],
            faceVertWorld0[:, 2, :] - faceVertWorld0[:, 0, :],
        )
        norm0 = np.linalg.norm(tmp0, ord=2, axis=1)
        faceNormalWorld0 = np.divide(tmp0, np.clip(norm0, a_min=1.e-6, a_max=np.inf)[:, None])

        # loaded things during training
        with open(projRoot + 'v/R/%s/R11a1/%08d.pkl' % (dataset, j), 'wb') as f:
            pickle.dump({
                'vertWorld0': vertWorld0,
                'face0': face0,
                'faceOpenrooms60o': faceOpenrooms60o,
                'faceObjID0': faceObjID0,
                'faceNormalWorld0': faceNormalWorld0,
            }, f)
        # This part is useless
        # with open(projRoot + 'v/R/%s/R11a2/%08d.pkl' % (dataset, j), 'wb') as f:
        #     pickle.dump({
        #         'vfso0List': vfso0List,
        #     }, f)

        if j in indChosenDump:
            cmap = get_cmap('inferno')
            dumpPly(visualDir + 'vertWorld_%s_%08d.ply' % (dataset, j), vertWorld0, face0)
            faceRgb0 = cmap(faceOpenrooms60o / 60.)[:, :3]
            dumpPly2(visualDir + 'semanticOpenrooms60_%s_%08d.ply' % (dataset, j),
                     vertWorld0, face0, faceRgb0)

            faceRgb0 = cmap(faceObjID0 / faceObjID0.max())[:, :3]
            dumpPly2(visualDir + 'objID_%s_%08d.ply' % (dataset, j),
                     vertWorld0, face0, faceRgb0)


if __name__ == '__main__':
    main()
