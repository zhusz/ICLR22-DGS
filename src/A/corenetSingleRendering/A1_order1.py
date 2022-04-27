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
from codes_py.np_ext.mat_io_v1 import pSaveMat
import pickle
import numpy as np


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
    corenetRawRoot = projRoot + 'remote_fastdata/corenet/data/'

    sptTag = 'single'
    count = 0
    sptTagList = []
    splitTagList = []
    corenetShaH3TagList = []
    corenetShaTagList = []
    for splitTag in ['train', 'val', 'test']:
        corenetShaH3Tags = sorted([x for x in os.listdir(corenetRawRoot + '%s.%s/' % (
            sptTag, splitTag
        )) if len(x) == 3])
        for corenetShaH3Tag in corenetShaH3Tags:
            print('Processing A1 for %s: Now Count is %d (sptTag: %s, splitTag: %s, corenetShaH3Tag: %s)' %
                  (dataset, count, sptTag, splitTag, corenetShaH3Tag))
            corenetShaTags = sorted([x[:-4] for x in os.listdir(corenetRawRoot + '%s.%s/%s/' %
                                                                (sptTag, splitTag, corenetShaH3Tag))
                                     if x.endswith('.npz')])
            for corenetShaTag in corenetShaTags:
                sptTagList.append(sptTag)
                splitTagList.append(splitTag)
                corenetShaH3TagList.append(corenetShaH3Tag)
                corenetShaTagList.append(corenetShaTag)
            count += len(corenetShaTags)

    m = count
    flagSplit = np.zeros((m, ), dtype=np.int32)
    for j in range(m):
        if splitTagList[j] == 'train':
            flagSplit[j] = 1
        elif splitTagList[j] == 'val':
            flagSplit[j] = 2
        elif splitTagList[j] == 'test':
            flagSplit[j] = 3
        else:
            print('[Error] Unknown flagSplit - %s. Please check' % (splitTagList[j]))
            import ipdb
            ipdb.set_trace()
            raise ValueError('You cannot proceed.')

    # read each npz
    camObj2World = np.zeros((m, 4, 4), dtype=np.float32)
    camWorld2Cam = np.zeros((m, 4, 4), dtype=np.float32)
    cameraTransform = np.zeros((m, 4, 4), dtype=np.float32)
    catIDList = []
    shaIDList = []
    for j in range(m):
        if j % 100 == 0 or j < 5:
            print('Process npz for %s A1: %d / %d' % (dataset, j, m))
        sptTag = sptTagList[j]
        splitTag = splitTagList[j]
        corenetShaH3Tag = corenetShaH3TagList[j]
        corenetShaTag = corenetShaTagList[j]
        d = dict(np.load(corenetRawRoot + '%s.%s/%s/%s.npz' % (
            sptTag, splitTag, corenetShaH3Tag, corenetShaTag,
        )))
        camObj2World[j] = d['mesh_object_to_world_transforms'][0, :, :]
        camWorld2Cam[j] = d['view_transform']
        cameraTransform[j] = d['camera_transform']
        catIDList.append(str(d['mesh_labels'][0]))
        shaIDList.append(str(d['mesh_filenames'][0]))

    with open(projRoot + 'v/A/%s/A1_order1.pkl' % dataset, 'wb') as f:
        pickle.dump({
            'm': m,
            'flagSplit': flagSplit,
            'sptTagList': sptTagList,
            'splitTagList': splitTagList,
            'corenetShaH3TagList': corenetShaH3TagList,
            'corenetShaTagList': corenetShaTagList,
            'camObj2World': camObj2World,
            'camWorld2Cam': camWorld2Cam,
            'cameraTransform': cameraTransform,
            'catIDList': catIDList,
            'shaIDList': shaIDList,
        }, f)


if __name__ == '__main__':
    main()
