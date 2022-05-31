# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (tfcodna)
import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
import imageio
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imsave
from shutil import copyfile


def main():
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
    hdr_root = projRoot + 'remote_fastdata/openrooms/Image/'
    assert os.path.isdir(projRoot + 'cache/')
    target_root = projRoot + 'cache/dataset/openrooms/Image/'
    os.makedirs(target_root, exist_ok=True)

    A1_house = udl('pkl_A1_', 'openrooms')
    m_house = A1_house['m']
    sceneList_house = A1_house['sceneList']
    xmlTagList_house = A1_house['xmlTagList']

    # tonemap1 = cv2.createTonemap(gamma=2.2)
    tonemapDict = {
        'vanilla': cv2.createTonemap(gamma=2.2),
        'drago': cv2.createTonemapDrago(gamma=2.2),
        'mantiuk': cv2.createTonemapMantiuk(gamma=2.2),
        'reinhard': cv2.createTonemapReinhard(gamma=2.2),
        # 'durand': cv2.createTonemapDurand(gamma=2.2),
    }
    for j_house in range(5):
        sceneTag = sceneList_house[j_house]
        xmlTag = xmlTagList_house[j_house]
        # only check view 1
        viewID = 1

        input_file_name = hdr_root + 'main_%s/%s/im_%d.hdr' % (xmlTag, sceneTag, viewID)
        im = imageio.imread(input_file_name)
        for tonemapMethodName in tonemapDict.keys():
            rgb = tonemapDict[tonemapMethodName].process(im.copy())
            rgb = np.clip(rgb, a_min=0, a_max=1)
            '''
            print('%s: min: %.3f, max: %.3f, count invalid %d' %
                  (input_file_name, rgb[np.isfinite(rgb)].min(), rgb[np.isfinite(rgb)].max(),
                   (np.isfinite(rgb) == 0).sum()))
            '''
            imsave(target_root + 'rgb_%s_%s_%d_%s.png' %
                   (xmlTag, sceneTag, viewID, tonemapMethodName), rgb)
        copyfile(
            input_file_name,
            target_root + 'rgb_%s_%s_%d.hdr' % (xmlTag, sceneTag, viewID))

    '''
    for j, fnSource in enumerate(allList):
        if j < 5 or j % 1000 == 0:
            print('Processing script_dumpImg for openrooms: %d / %d' % (j, len(allList)))
        assert fnSource.endswith('.hdr')
        im = imageio.imread(hdr_root + fnSource, format='HDR-FI')

        # plt.imshow(np.clip(im, 0, 1))
        # plt.show()
        import ipdb
        ipdb.set_trace()
        print(1 + 1)
    '''

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()
