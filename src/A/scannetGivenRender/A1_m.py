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
import numpy as np
from codes_py.np_ext.np_image_io_v1 import imread
import pickle


bt = lambda s: s[0].upper() + s[1:]


def wrong():
    print('Problem occurred. Please check!')
    import ipdb
    ipdb.set_trace()
    print(1 + 1)


def checkIntrinsics(**kwargs):
    sceneRoot = kwargs['sceneRoot']

    out = dict()

    fns = os.listdir(sceneRoot + 'intrinsic/')
    assert len(fns) == 4
    for a in ['extrinsic', 'intrinsic']:
        for b in ['color', 'depth']:
            assert '%s_%s.txt' % (a, b) in fns
    for b in ['color', 'depth']:
        # "extrinsics"
        tmp = np.loadtxt(sceneRoot + 'intrinsic/extrinsic_%s.txt' % b, delimiter=' ')
        if not np.all(tmp == np.eye(4)):
            wrong()
        # intrinsics
        tmp = np.loadtxt(sceneRoot + 'intrinsic/intrinsic_%s.txt' % b, delimiter=' ')
        for r, c in [(0, 3), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)]:
            if not tmp[r, c] == 0:
                wrong()
        for r, c in [(2, 2), (3, 3)]:
            if not tmp[r, c] == 1:
                wrong()
        # if not tmp[0, 0] == tmp[1, 1]:
        #     wrong()
        out['%s_focalLengthWidth' % b] = float(tmp[0, 0])
        out['%s_focalLengthHeight' % b] = float(tmp[1, 1])
        out['%s_ppWidth' % b] = float(tmp[0, 2])
        out['%s_ppHeight' % b] = float(tmp[1, 2])

    return out


def checkColorDepthPoses(sceneRoot):
    out = dict()

    fns_color = os.listdir(sceneRoot + 'color/')
    fns_depth = os.listdir(sceneRoot + 'depth/')
    fns_pose = os.listdir(sceneRoot + 'pose/')
    if not (len(fns_color) == len(fns_depth) == len(fns_pose)):
        wrong()
    nFrame = len(fns_color)
    for t in range(nFrame):
        if '%d.jpg' % t not in fns_color: wrong()
        if '%d.png' % t not in fns_depth: wrong()
        if '%d.txt' % t not in fns_pose: wrong()
    rawMat = np.zeros((nFrame, 4, 4), dtype=np.float32)
    for t in range(nFrame):
        tmp = np.loadtxt(sceneRoot + 'pose/%d.txt' % t, delimiter=' ')
        rawMat[t, :, :] = tmp
    out['nFrame'] = nFrame
    out['rawPoseMat'] = rawMat

    color0 = imread(sceneRoot + 'color/0.jpg')
    depth0 = imread(sceneRoot + 'depth/0.png', required_dtype='uint16', required_nc=1)
    out['color_winWidth'] = color0.shape[1]
    out['color_winHeight'] = color0.shape[0]
    out['depth_winWidth'] = depth0.shape[1]
    out['depth_winHeight'] = depth0.shape[0]

    return out


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'

    dataset_house = 'scannet'
    A1_house = udl('_A1_', dataset_house)
    m_house = A1_house['m']
    scanIDList_house = A1_house['scanIDList']
    fileList_house = A1_house['fileList']

    dataset_cache_root = projRoot + 'remote_fastdata/cache/scannet/'

    fxywxyColor = []
    fxywxyDepth = []
    ppxyColor = []
    ppxyDepth = []
    rawPoseMat = []
    houseIDList = []
    viewIDList = []

    for j_house in range(m_house):
        print('Processing A1_m for %s: %d (j_house) / %d (m_house)' %
              (dataset, j_house, m_house))
        sceneRoot = dataset_cache_root + fileList_house[j_house]

        # intrinsics
        intrinsics = checkIntrinsics(sceneRoot=sceneRoot)

        # all the others
        others = checkColorDepthPoses(sceneRoot=sceneRoot)

        nFrame = others['nFrame']
        fxywxyColor.append(np.tile(np.array(
            [intrinsics['color_focalLengthWidth'], intrinsics['color_focalLengthHeight'],
             others['color_winWidth'], others['color_winHeight']], dtype=np.float32
        )[None, :], (nFrame, 1)))
        fxywxyDepth.append(np.tile(np.array(
            [intrinsics['depth_focalLengthWidth'], intrinsics['depth_focalLengthHeight'],
             others['depth_winWidth'], others['depth_winHeight']], dtype=np.float32
        )[None, :], (nFrame, 1)))
        ppxyColor.append(np.tile(np.array(
            [intrinsics['color_ppWidth'], intrinsics['color_ppHeight']], dtype=np.float32
        )[None, :], (nFrame, 1)))
        ppxyDepth.append(np.tile(np.array(
            [intrinsics['depth_ppWidth'], intrinsics['depth_ppHeight']], dtype=np.float32
        )[None, :], (nFrame, 1)))
        rawPoseMat.append(others['rawPoseMat'])

        houseIDList.append(j_house * np.ones((others['nFrame'], ), dtype=np.int32))
        viewIDList.append(np.arange(others['nFrame']).astype(np.int32))

    fxywxyColor_ = np.concatenate(fxywxyColor, 0)
    fxywxyDepth_ = np.concatenate(fxywxyDepth, 0)
    ppxyColor_ = np.concatenate(ppxyColor, 0)
    ppxyDepth_ = np.concatenate(ppxyDepth, 0)
    rawPoseMat_ = np.concatenate(rawPoseMat, 0)
    houseIDList = np.concatenate(houseIDList)
    viewIDList = np.concatenate(viewIDList)
    m = fxywxyColor_.shape[0]

    with open(projRoot + 'v/A/%s/A1_m.pkl' % dataset, 'wb') as f:
        pickle.dump({
            'm': m,
            'houseDataset': dataset_house,
            'fxywxyColor': fxywxyColor_,
            'fxywxyDepth': fxywxyDepth_,
            'ppxyColor': ppxyColor_,
            'ppxyDepth': ppxyDepth_,
            'rawPoseMat': rawPoseMat_,
            'houseIDList': houseIDList,
            'viewIDList': viewIDList,
        }, f)


if __name__ == '__main__':
    main()
