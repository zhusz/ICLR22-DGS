# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP
from codes_py.corenet.geometry.voxelization import voxelize_mesh
from codes_py.corenet.cc import fill_voxels
from codes_py.toolbox_3D.mesh_surgery_v1 import trimVert
from codes_py.toolbox_3D.representation_v1 import voxSdfSign2mesh_skmc
from codes_py.toolbox_3D.mesh_io_v1 import dumpPly


def voxelizeMesh(ifAttemptShoot, vert0, face0, goxyz, sCell, minBound0, maxBound0, Lx, Ly, Lz, cudaDevice):
    # ifAttemptShoot: in the case of a whole house voxelization, it is likely that
    # the whole room is labeled as occupied as it is a closed room.
    # In that scenario, if ifAttemptShoot is enabled, then we will stop this from happening,
    # where the voxels being hit by the shooting (ray) will be labeled as not occupied
    # before internal filing.
    # Typically, for object voxelization, this ifAttemptShoot will not be set to True.

    # Note min/maxBound0 can be slacked
    # so cannot be determined only by goxyz/sCell
    vertZO0 = (vert0 - minBound0[None, :]) / (maxBound0 - minBound0)[None, :]
    faceVertZO0 = vertInfo2faceVertInfoNP(vertZO0[None], face0[None])[0]
    grids_thcpu = voxelize_mesh(
        torch.from_numpy(faceVertZO0),
        torch.from_numpy(np.array([int(faceVertZO0.shape[0])], dtype=np.int32)),
        resolution=(Lx, Ly, Lz),
        view2voxel=torch.from_numpy(
            np.array([
                [Lx, 0, 0, 0],
                [0, Ly, 0, 0],
                [0, 0, Lz, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32),
        ),
        sub_grid_sampling=False,
        image_resolution_multiplier=8,
        conservative_rasterization=False,
        projection_depth_multiplier=1,
        cuda_device=None,
    )
    grids_thgpu = grids_thcpu.to(cudaDevice)
    fill_voxels.fill_inside_voxels_gpu(grids_thgpu, inplace=True)
    occupancy_ratio = float(grids_thgpu.sum()) / float(grids_thgpu.numel())

    if ifAttemptShoot and occupancy_ratio > 0.3:
        # Here assume [0, 1, 0] is the U direction
        halfX = int(Lx / 2.)
        halfZ = int(Lz / 2.)
        grids_thcpu[0, halfZ, :, halfX] = 0
        grids_thgpu = grids_thcpu.to(cudaDevice)
        fill_voxels.fill_inside_voxels_gpu(grids_thgpu, inplace=True)
        final_occupancy_ratio = float(grids_thgpu.sum()) / float(grids_thgpu.numel())
        ifShooted = True
    else:
        final_occupancy_ratio = occupancy_ratio
        ifShooted = False

    return grids_thgpu[0].detach().cpu().numpy(), grids_thcpu[0].detach().cpu().numpy(), \
        occupancy_ratio, final_occupancy_ratio, ifShooted


# voxel label determination rule
#   Assumes all the query points are "nonsurface". Nearsurface point should be determined
#       in a different way (using the mesh)
#   If grid0 indicates a nonsurface point is not occupied, then it is just not occupied
#   If grid0 indicates a nonsurface point is occupied, then search through all the object
#       partGrid0, and determine whether it is occupied.
#       If none of the object indicates it is occupied, then it is unoccupied.
#   Note grid0 is more likely to say a point is occupied, as the space being delineated by
#       multiple objects will be filled up. In our implementation, these space will be
#       treated as unoccupied - this level of noise won't affect the overall accuracy.

# No we still need to care about the nearsurface in voxels, as the mesh is non-oriented.
# So the solution is: you need to label out all the voxels in grid0 that is a boundary voxels
# and shall always need to go through the per-object check.

def main():
    cudaDevice = 'cuda:0'

    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))

    rawRoot = projRoot + 'remote_fastdata/openrooms/'
    visualDir = projRoot + 'cache/dataset/%s/dump_R11b_vox/' % dataset
    os.makedirs(visualDir, exist_ok=True)

    A1 = udl('pkl_A1_', dataset)
    m = A1['m']
    sceneList = A1['sceneList']
    xmlTagList = A1['xmlTagList']
    del A1

    indChosenDump = list(range(5)) + [46]

    L = 128
    Lx, Ly, Lz = L, L, L
    slackRate = 0.01

    partL = 64
    partLx, partLy, partLz = partL, partL, partL
    partSlackRate = 0.01

    for j in range(m):
        if j in [502, 522, 952, 1084, 1178, 1479, 1790, 2240]:
            continue
        print('Processing R11b_vox for %s: %d / %d' % (dataset, j, m))

        A11a1 = udl('pkls_R11a1_', dataset, int(j))
        # A11a2 = udl('pkls_R11a2_', dataset, int(j))

        vertWorld0 = A11a1['vertWorld0']
        face0 = A11a1['face0']
        faceOpenrooms60o = A11a1['faceOpenrooms60o']
        faceObjID0 = A11a1['faceObjID0']

        minBoundWorld0 = vertWorld0.min(0)
        maxBoundWorld0 = vertWorld0.max(0)
        minBoundWorld0, maxBoundWorld0 = \
            (1 + slackRate) * minBoundWorld0 + (-slackRate) * maxBoundWorld0, \
            (-slackRate) * minBoundWorld0 + (1 + slackRate) * maxBoundWorld0
        sCell = (maxBoundWorld0 - minBoundWorld0) / np.array([Lx, Ly, Lz], dtype=np.float32)
        goxyz = minBoundWorld0 + 0.5 * sCell
        grid0, nonfilledGrid0, occupancy_ratio, final_occupancy_ratio, ifShooted = voxelizeMesh(
            True,
            vertWorld0, face0, goxyz, sCell, minBoundWorld0, maxBoundWorld0,
            Lx, Ly, Lz, cudaDevice
        )
        if ifShooted:
            print('    Shoot happens: original occupancy rate: %.3f, now occupancy rate: %.3f' %
                  (occupancy_ratio, final_occupancy_ratio))
        # Debugging purpose - does grid0 contain more occupied voxels? The answer is yes.
        # partGrid0 = np.zeros_like(grid0)
        # for o in np.unique(faceObjID0):
        #     flag0 = (faceObjID0 == o)
        #     partVertWorld0, partFace0 = trimVert(vertWorld0, face0[flag0])

        #     partGrid0_current, _ = voxelizeMesh(
        #         partVertWorld0, partFace0, goxyz, sCell, minBoundWorld0, maxBoundWorld0,
        #         Lx, Ly, Lz, cudaDevice
        #     )
        #     partGrid0[partGrid0_current > 0] = 1

        # # debug visualization
        # v0, f0 = voxSdfSign2mesh_skmc(
        #     (1 - grid0).transpose((1, 2, 0)), goxyz, sCell)
        # v1, f1 = voxSdfSign2mesh_skmc(
        #     (1 - (grid0 - partGrid0)).transpose((1, 2, 0)), goxyz, sCell)
        # dumpPly(visualDir + '%s_%d_0.ply' % (dataset, j), v0, f0)
        # dumpPly(visualDir + '%s_%d_1.ply' % (dataset, j), v1, f1)

        nObj = len(np.unique(faceObjID0))
        partsGrid0 = np.zeros((nObj, partLz, partLy, partLx), dtype=bool)
        openrooms60PerObj = -np.ones((nObj, ), dtype=np.int32)
        goxyzPerObj = np.zeros((nObj, 3), dtype=np.float32)
        sCellPerObj = np.zeros((nObj, 3), dtype=np.float32)
        for o in range(nObj):
            flag0 = (faceObjID0 == o)
            assert flag0.sum() > 0
            partVertWorld0, partFace0 = trimVert(vertWorld0, face0[flag0])
            partFaceOpenrooms60o = faceOpenrooms60o[flag0]
            assert len(np.unique(partFaceOpenrooms60o)) == 1
            openrooms60PerObj[o] = int(partFaceOpenrooms60o[0])
            partMinBoundWorld0 = partVertWorld0.min(0)
            partMaxBoundWorld0 = partVertWorld0.max(0)
            partMinBoundWorld0, partMaxBoundWorld0 = \
                (1 + partSlackRate) * partMinBoundWorld0 + (-partSlackRate) * partMaxBoundWorld0, \
                (-partSlackRate) * partMinBoundWorld0 + (1 + partSlackRate) * partMaxBoundWorld0
            partSCell = (partMaxBoundWorld0 - partMinBoundWorld0) / \
                np.array([partLx, partLy, partLz], dtype=np.float32)
            partGoxyz = partMinBoundWorld0 + 0.5 * partSCell
            tmp0, _, _, _, _ = voxelizeMesh(
                False,
                partVertWorld0, partFace0, partGoxyz, partSCell, partMinBoundWorld0, partMaxBoundWorld0,
                partLx, partLy, partLz, cudaDevice
            )
            partsGrid0[o, :, :, :] = tmp0
            goxyzPerObj[o, :] = partGoxyz
            sCellPerObj[o, :] = partSCell

        r = 1
        d = 3
        kernelForNeighbouring = np.ones((1, 1, d, d, d), dtype=np.float32)
        kernelForNeighbouring_thgpu = torch.from_numpy(kernelForNeighbouring).to(cudaDevice)
        checkbook_thgpu = F.conv3d(
            input=torch.from_numpy(grid0[None, None, :, :, :]).to(cudaDevice),
            weight=kernelForNeighbouring_thgpu, bias=None, stride=1, padding=r,
        )
        checkbook0 = (checkbook_thgpu[0, 0, :, :, :].detach().cpu().numpy() > 0.5)

        packedCheckbook0 = np.packbits(checkbook0.reshape((Lx * Ly * Lz)), bitorder='big')
        packedPartsGrid0 = np.packbits(partsGrid0.reshape((
            nObj, partLx, partLy, partLz)), bitorder='big')
        np.savez_compressed(
            projRoot + 'v/R/%s/R11b/%08d.npz' % (dataset, j),
            packedCheckbook0=packedCheckbook0,
            checkbookGoxyz=goxyz, checkbookSCell=sCell,
            checkbookLx=Lx, checkbookLy=Ly, checkbookLz=Lz,
            ifShooted=ifShooted, occupancy_ratio=occupancy_ratio,
            final_occupancy_ratio=final_occupancy_ratio,
            packedPartsGrid0=packedPartsGrid0,
            partsGoxyz=goxyzPerObj, partsSCell=sCellPerObj,
            nObj=nObj, partLx=partLx, partLy=partLy, partLz=partLz,
            partsOpenrooms60=openrooms60PerObj,
        )
        # Notes all occupancy in here is 1 and unoccupancy is 0.
        # Usage: First check checkbook, if it is occupied (1), then iterate overall the objects
        # It is likely that checkbook returns occupied (1), but none of the objects indicates
        # occupied. Checkbook is a overly recalled book, so it is called checkbook.

        # visualization
        if j in indChosenDump:
            grid_v0, grid_f0 = voxSdfSign2mesh_skmc(
                (1 - grid0).transpose((1, 2, 0)), goxyz, sCell
            )
            dumpPly(visualDir + '%s_%d_grid0.ply' % (dataset, j), grid_v0, grid_f0)
            checkbook_v0, checkbook_f0 = voxSdfSign2mesh_skmc(
                (1 - checkbook0.astype(np.float32)).transpose((1, 2, 0)), goxyz, sCell
            )
            dumpPly(visualDir + '%s_%d_checkbook0.ply' % (dataset, j),
                    checkbook_v0, checkbook_f0)
            nonfilled_v0, nonfilled_f0 = voxSdfSign2mesh_skmc(
                (1 - nonfilledGrid0).transpose((1, 2, 0)), goxyz, sCell
            )
            dumpPly(visualDir + '%s_%d_nonfilled0.ply' % (dataset, j),
                    nonfilled_v0, nonfilled_f0)
            for o in range(nObj):
                obj_v0, obj_f0 = voxSdfSign2mesh_skmc(
                    (1 - partsGrid0[o].astype(np.float32)).transpose((1, 2, 0)),
                    goxyzPerObj[o, :], sCellPerObj[o, :],
                )
                dumpPly(visualDir + '%s_%d_obj%d.ply' % (dataset, j, o), obj_v0, obj_f0)


if __name__ == '__main__':
    main()

