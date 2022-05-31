# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from codes_py.np_ext.mat_io_v1 import pSaveMat
import pickle


# 1. Layout Mesh:
# https://drive.google.com/file/d/1OOc_55C0YkEPC_s81i0s8jf8xEp1v3YR/view?usp=sharing
# 2. Object Mesh:
# https://drive.google.com/file/d/1l8nyY8bgslPqz41e1nWPTJTWNockqIQV/view?usp=sharing
# 3. Scene configuration:
# https://drive.google.com/file/d/1d2CTLZeE1vya8mf0S0V8NDXGEak4Rx02/view?usp=sharing
# 4. Rendered image:
# https://drive.google.com/file/d/11046xQ9L6SkahoxF0W3cn__pmFZQYKyv/view?usp=sharing
def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
    rawRoot = projRoot + 'remote_fastdata/openrooms/'
    layoutList = sorted([x for x in os.listdir(rawRoot + 'layoutMesh/') if x.startswith('scene')])

    xmlSceneList = sorted([x for x in os.listdir(rawRoot + 'scenes/xml/') if x.startswith('scene')])
    xml1SceneList = sorted([x for x in os.listdir(rawRoot + 'scenes/xml1/') if x.startswith('scene')])

    assert len(xmlSceneList) == len(xml1SceneList)
    for j in range(len(xmlSceneList)):
        assert xmlSceneList[j] == xml1SceneList[j]
        assert xmlSceneList[j] in layoutList
    sceneList = xmlSceneList

    m = len(xmlSceneList) * 2
    sceneList = sceneList * 2
    xmlTagList = ['xml'] * len(xmlSceneList) + ['xml1'] * len(xmlSceneList)

    with open(projRoot + 'v/A/%s/A1_order.pkl' % dataset, 'wb') as f:
        pickle.dump({
            'm': m,
            'sceneTagList': sceneList,
            'xmlTagList': xmlTagList,
        }, f)
    pSaveMat(
        projRoot + 'v/A/%s/A1_order.mat' % dataset,
        {
            'm': m,
            'sceneTagList': sceneList,
            'xmlTagList': xmlTagList,
        }
    )

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()
