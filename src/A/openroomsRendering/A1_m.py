# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# header files both in pkl and mat

import os
import sys
import pickle
import numpy as np

projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
from codes_py.np_ext.mat_io_v1 import pSaveMat


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    dataset_house = 'openrooms'
    A1_house = udl('pkl_A1_', dataset_house)
    m_house = A1_house['m']
    xmlTagList_house = A1_house['xmlTagList']
    sceneTagList_house = A1_house['sceneTagList']

    houseIDList = []
    viewTagList = []
    hdr_image_folder = projRoot + 'remote_fastdata/openrooms/Image/'
    for j_house in range(m_house):
        print('Processing A1 for %s: %d (j_house) / %d (m_house)' %
              (dataset, j_house, m_house))
        xmlTag = xmlTagList_house[j_house]
        sceneTag = sceneTagList_house[j_house]
        current_folder = hdr_image_folder + 'main_%s/%s/' % (xmlTag, sceneTag)
        if not os.path.isdir(current_folder):
            continue
        fns = sorted(os.listdir(current_folder))
        for k in range(len(fns)):
            assert fns[k].startswith('im_')
            assert fns[k].endswith('.hdr')
            assert 0 <= int(fns[k][3:-4]) <= len(fns)
        houseIDList.append(j_house * np.ones((len(fns), ), dtype=np.int32))
        viewTagList.append(1 + np.arange(len(fns), dtype=np.int32))

    houseIDList = np.concatenate(houseIDList, 0)
    viewTagList = np.concatenate(viewTagList, 0)
    m = int(houseIDList.shape[0])

    with open(projRoot + 'v/A/%s/A1_order1.pkl' % dataset, 'wb') as f:
        pickle.dump({
            'm': m,
            'houseIDList': houseIDList,
            'viewTagList': viewTagList,
            'houseDataset': dataset_house,
        }, f)
    pSaveMat(projRoot + 'v/A/%s/A1_order1.mat' % dataset, {
        'm': m,
        'houseIDList': houseIDList,
        'viewTagList': viewTagList,
    })

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()
