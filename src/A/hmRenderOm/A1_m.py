# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# (tfconda)
from collections import OrderedDict
from easydict import EasyDict
import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
from codes_py.toolbox_3D.rotations_v1 import quat2matNP, ELU02cam0
from codes_py.toolbox_3D.mesh_io_v1 import load_obj_np
from codes_py.toolbox_show_draw.html_v1 import HTMLStepper
from codes_py.toolbox_show_draw.draw_v1 import to_heatmap
import math
import numpy as np
import json
from skimage.io import imread
import trimesh
import pickle
from codes_py.toolbox_3D.pyrender_wrapper_v1 import PyrenderManager


def main():
    dataset = os.path.realpath(__file__).split('/')[-2]
    rgb_raw_root = projRoot + 'remote_fastdata/omnidata/om/rgb/hm3d/'  # subject to change
    point_info_raw_root = projRoot + 'remote_fastdata/omnidata/om/point_info/hm3d/'  # subject to change
    zbuffer_raw_root = projRoot + 'remote_fastdata/omnidata/om/depth_zbuffer/hm3d/'

    dataset_house = 'hm'
    assert dataset.startswith(dataset_house)
    A1_house = udl('pkl_A1_', dataset_house)
    m_house = A1_house['m']
    houseShaList_house = A1_house['houseShaList']
    flagSplit_house = udl('_A01_flagSplit', dataset_house)

    houseIDList = []
    pointIDList = []
    viewIDList = []
    EWorld = []
    LWorld = []
    UWorld = []
    preliminaryFlagSplit = []
    resolution = []
    fScaleList = []

    assert os.path.isdir(projRoot + 'cache/')
    visualDir = projRoot + 'cache/dataset/%s/A1/' % dataset
    os.makedirs(visualDir, exist_ok=True)
    htmlStepper = HTMLStepper(visualDir, 100, 'A1')

    indChosen_house = list(range(m_house))
    # indVisChosen_house = [0, 1, 100, 200, 500, 850]
    indVisChosen_house = list(range(10))
    pyrenderManager = PyrenderManager(512, 512)

    for j_house in indChosen_house:
        print('Processing A1 for %s: %d (j_house) / %d (m_house)' %
              (dataset, j_house, m_house))

        curr_rgb_raw_dir = rgb_raw_root + '%05d-' % j_house + \
                           houseShaList_house[j_house] + '/'
        curr_zbuffer_raw_dir = zbuffer_raw_root + '%05d-' % j_house + \
                               houseShaList_house[j_house] + '/'
        curr_point_info_raw_dir = point_info_raw_root + '%05d-' % j_house + \
                                  houseShaList_house[j_house] + '/'
        fns = os.listdir(curr_rgb_raw_dir)
        ps = np.array([int(fn.split('_')[1]) for fn in fns], dtype=np.int32)
        vs = np.array([int(fn.split('_')[3]) for fn in fns], dtype=np.int32)
        cs = np.sort_complex(ps + 1j * vs)
        ps = np.real(cs).astype(np.int32)
        vs = np.imag(cs).astype(np.int32)

        nShot = len(fns)
        ECurrent = np.zeros((nShot, 3), dtype=np.float32)
        quatCurrent = np.zeros((nShot, 4), dtype=np.float32)
        fovRadCurrent = np.zeros((nShot, ), dtype=np.float32)
        resolutionCurrent = np.zeros((nShot, ), dtype=np.float32)
        prelimilaryFlagSplitCurrent = flagSplit_house[j_house] * np.ones((nShot, ), dtype=np.int32)

        for i, (pid, vid) in enumerate(zip(ps, vs)):
            json_fn = curr_point_info_raw_dir + 'point_%d_view_%d_domain_fixatedpose.json' % \
                      (pid, vid)
            if os.path.isfile(json_fn):
                with open(json_fn, 'r') as f:
                    camInfo = json.load(f)
                ECurrent[i, :] = camInfo['camera_location']
                quatCurrent[i, :] = camInfo['camera_rotation_final_quaternion']
                fovRadCurrent[i] = camInfo['field_of_view_rads']
                resolutionCurrent[i] = camInfo['resolution']
            else:
                prelimilaryFlagSplitCurrent[i] = 0
        rotMatCurrent = quat2matNP(quatCurrent)
        ELCurrent = -(
            np.concatenate([
                np.zeros((nShot, 2), dtype=np.float32),
                np.ones((nShot, 1), dtype=np.float32),
            ], 1)[:, None, :]  # (nShot, 1, 3)
            * rotMatCurrent  # (nShot, 3, 3)
        ).sum(2)
        ELCurrent = ELCurrent / \
                    np.clip(np.linalg.norm(ELCurrent, axis=1, ord=2),
                            a_min=1.e-4, a_max=np.inf)[:, None]
        LCurrent = ECurrent + ELCurrent
        UCurrent = -(
            np.concatenate([
                np.zeros((nShot, 1), dtype=np.float32),
                -np.ones((nShot, 1), dtype=np.float32),
                np.zeros((nShot, 1), dtype=np.float32),
            ], 1)[:, None, :]  # (nShot, 1, 3)
            * rotMatCurrent  # (nShot, 3, 3)
        ).sum(2)

        if nShot > 0:
            houseIDList.append(j_house * np.ones((nShot, ), dtype=np.int32))
            pointIDList.append(ps.astype(np.int32))
            viewIDList.append(vs.astype(np.int32))
            EWorld.append(ECurrent.astype(np.float32))
            LWorld.append(LCurrent.astype(np.float32))
            UWorld.append(UCurrent.astype(np.float32))
            resolution.append(resolutionCurrent.astype(np.int32))
            fScaleCurrent = 1. / np.tan(fovRadCurrent / 2.)
            fScaleList.append(fScaleCurrent)
            preliminaryFlagSplit.append(prelimilaryFlagSplitCurrent.astype(np.int32))

        if j_house in indVisChosen_house:
            summary0 = OrderedDict([])
            txt0 = []
            brInds = [0, ]

            chosenK = [100, 200, 300, 400, 500, 600, 700, 800]
            obj_file_name = projRoot + 'remote_fastdata/hm/install/%05d-%s/%s.obj' % \
                            (j_house, houseShaList_house[j_house], houseShaList_house[j_house])
            assert os.path.isfile(obj_file_name), obj_file_name
            vertWorld0, face0 = load_obj_np(obj_file_name)

            for k in chosenK:
                omnidata_img = imread(curr_rgb_raw_dir + 'point_%d_view_%d_domain_rgb.png' % (ps[k], vs[k]))
                summary0['omnidata_img_%d_%d' % (ps[k], vs[k])] = omnidata_img
                txt0.append('k %d Point %d View %d img' % (k, ps[k], vs[k]))

                omnidata_depth = imread(curr_zbuffer_raw_dir + 'point_%d_view_%d_domain_depth_zbuffer.png' % (ps[k], vs[k]))

                omnidata_depth = omnidata_depth.astype(np.float32) / 512.  # only for omnidata-hm3d
                # other datasets under omnidata is 8000 for this one.

                summary0['omnidata_depthZBuffer_%d_%d' % (ps[k], vs[k])] = to_heatmap(omnidata_depth, vmin=0.5, vmax=6, cmap='inferno')
                txt0.append('k %d Point %d View %d depthZBuffer' % (k, ps[k], vs[k]))

                winSize = resolutionCurrent[k]
                # fScale = 1. / math.tan(fovRadCurrent[k] / 2.)
                fScale = fScaleCurrent[k]
                focalLength = fScale * winSize / 2.
                assert winSize == 512
                pyrenderManager.clear()
                pyrenderManager.add_plain_mesh(vertWorld0, face0)
                pyrenderManager.add_camera(
                    np.array([focalLength, focalLength, winSize, winSize], dtype=np.float32),
                    np.linalg.inv(ELU02cam0(np.concatenate([
                        ECurrent[k, :], LCurrent[k, :], UCurrent[k, :],
                    ], 0)))
                )
                tmp = pyrenderManager.render()
                ratio = omnidata_depth / tmp[1]
                print('The ratio is Min: %.6f, Max: %.6f, Mean: %.6f' %
                      (ratio[np.isfinite(ratio)].min(),
                       ratio[np.isfinite(ratio)].max(),
                       ratio[np.isfinite(ratio)].mean(),
                       ))
                summary0['omnidata_renderZBuffer_%d_%d' % (ps[k], vs[k])] = to_heatmap(tmp[1], vmin=0.5, vmax=6, cmap='inferno')
                txt0.append('k %d Point %d View %d renderZBuffer min %.3f max %.3f mean %.3f'
                            % (k, ps[k], vs[k], ratio[np.isfinite(ratio)].min(),
                               ratio[np.isfinite(ratio)].max(),
                               ratio[np.isfinite(ratio)].mean()))
                brInds.append(len(summary0))

            htmlStepper.step2(
                summary0, txt0, brInds,
                headerMessage='HM3D %05d-%s' % (j_house, houseShaList_house[j_house]),
                subMessage='',
            )

    houseIDList = np.concatenate(houseIDList, 0)
    pointIDList = np.concatenate(pointIDList, 0)
    viewIDList = np.concatenate(viewIDList, 0)
    EWorld = np.concatenate(EWorld, 0)
    LWorld = np.concatenate(LWorld, 0)
    UWorld = np.concatenate(UWorld, 0)
    resolution = np.concatenate(resolution, 0)
    fScaleList = np.concatenate(fScaleList, 0)
    preliminaryFlagSplit = np.concatenate(preliminaryFlagSplit, 0)
    preliminaryFlagSplit[houseIDList == 887] = 0  # rgb meaningless
    m = int(houseIDList.shape[0])

    # with open(projRoot + 'v/A/%s/A1_order1.pkl' % dataset, 'wb') as f:
    #     pickle.dump({
    #         'm': m,
    #         'houseIDList': houseIDList,
    #         'pointIDList': pointIDList,
    #         'viewIDList': viewIDList,
    #         'EWorld': EWorld,
    #         'LWorld': LWorld,
    #         'UWorld': UWorld,
    #         'preliminaryFlagSplit': preliminaryFlagSplit,
    #         'resolution': resolution,
    #         'fScaleList': fScaleList,
    #         'houseDataset': 'hm',
    #     }, f)

    print(m)
    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()
