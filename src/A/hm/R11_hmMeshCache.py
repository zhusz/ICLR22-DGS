# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# (tfconda)
import torch
import pickle
import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
from codes_py.toolbox_3D.mesh_io_v1 import load_obj_np, dumpPly2
import numpy as np
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP


def main():
    hm3d_raw_root = projRoot + 'remote_fastdata/hm/install/'

    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    assert dataset == 'hm'
    A1 = udl('pkl_A1_', dataset)
    m = A1['m']   
    houseShaList = A1['houseShaList']
    zNear = 1.e-6

    indVisChosen = [0, 1, 800]
    assert os.path.isdir(projRoot + 'cache/')
    visualDir = projRoot + 'cache/dataset/%s/R11/' % (dataset)
    os.makedirs(visualDir, exist_ok=True)

    j1 = int(os.environ['J1'])
    j2 = int(os.environ['J2'])
    indChosen = list(range(j1, j2))

    for j in indChosen:
        print('Processing R11_hmMeshCache for %s: %d / %d' % (dataset, j, m))

        vertWorld0, face0 = load_obj_np(hm3d_raw_root + '%05d-%s/%s.obj' % 
            (j, houseShaList[j], houseShaList[j]))
        vertWorld0 = vertWorld0.astype(np.float32)
        face0 = face0.astype(np.int32)
        faceVertWorld0 = vertInfo2faceVertInfoNP(vertWorld0[None], face0[None])[0]       
        tmp0 = np.cross(
            faceVertWorld0[:, 1, :] - faceVertWorld0[:, 0, :],
            faceVertWorld0[:, 2, :] - faceVertWorld0[:, 0, :],
        )
        norm0 = np.linalg.norm(tmp0, axis=1, ord=2)
        tmp0 = tmp0 / np.clip(norm0, a_min=zNear, a_max=np.inf)[:, None]
        faceNormal0 = tmp0
        faceArea0 = norm0 / 2.

        with open(projRoot + 'v/R/%s/R11/%08d.pkl' % (dataset, j), 'wb') as f:
            pickle.dump({
                'vertWorld0': vertWorld0,
                'face0': face0,
                'faceNormalWorld0': faceNormal0,
                'faceArea0': faceArea0,
            }, f)
        
        if j in indVisChosen:
            dumpPly2(
                visualDir + '%05d-%s-faceNormal.ply' % (j, houseShaList[j]),
                vertWorld0, face0, np.abs(faceNormal0),
            )


if __name__ == '__main__':
    main()
