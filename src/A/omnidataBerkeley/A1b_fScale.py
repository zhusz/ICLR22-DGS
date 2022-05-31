# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (tfconda)
from email.contentmanager import raw_data_manager
import numpy as np
import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl


def main():
    dataset = os.path.realpath(__file__).split('/')[-2]
    omnidata_raw_root = projRoot + 'remote_fastdata/omnidata/'

    A1 = udl('pkl_A1_', dataset)
    m = A1['m']
    componentNameList = A1['componentNameList']
    componentIDList = A1['componentIDList']
    buildingNameList = A1['buildingNameList']
    pointIDList = A1['pointIDList']
    viewIDList = A1['viewIDList']
    flagSplit = A1['flagSplit']  # potentially filter out "point_info" non-exist cases

    # Choose your own indChosen in some other form
    indChosen = []
    for componentID in range(5):
        indChosen.append(np.argmax(componentIDList == componentID))

    fScaleWidthHeight = np.zeros((m, 2), dtype=np.float32)  # This is the original data
    ppxy = np.zeros((m, 4), dtype=np.float32)  # This is the original data
    fxywxy = np.zeros((m, 4), dtype=np.float32)  # The wxy part is the original data, while the fxy part is computed from fScaleWidthHeight and wxy
    fileMissingFlag = np.zeros((m, ), dtype=bool)

    for j in indChosen:
        print('Processing A1b_fScale for %s: %d / %d' % (dataset, j, m))
        componentName = componentNameList[j]
        buildingName = buildingNameList[j]
        pointID = int(pointIDList[j])
        viewID = int(viewIDList[j])
        partitionName = 'ta' if componentName == 'taskonomy' else 'om'
        domainName = 'point_info' if componentName in ['taskonomy', 'hypersim'] else 'fixatedpose'

        rgb_file = omnidata_raw_root + '%s/rgb/%s/%s/point_%d_view_%d_domain_rgb.png' % \
            (partitionName, componentName, buildingName, pointID, viewID)
        depth_file = omnidata_raw_root + '%s/depth_zbuffer/%s/%s/point_%d_view_%d_domain_depth_zbuffer.png' % \
            (partitionName, componentName, buildingName, pointID, viewID)
        json_file = omnidata_raw_root + '%s/%s/%s/%s/point_%d_view_%d_domain_%s.json' % \
            (partitionName, domainName, componentName, buildingName, pointID, viewID, domainName)
        if not (os.path.isfile(rgb_file) and os.path.isfile(depth_file) and os.path.isfile(json_file)):
            fileMissingFlag[j] = True           
            print('[WARNING] fileMissingFlag activated on %s-%d' % (dataset, j))
            continue
        
        # json
        if componentName in ['taskonomy', 'replica', 'gso']:
            import ipdb
            ipdb.set_trace()
            print(1 + 1)
            TODO  # field_of_rads
        elif componentName in ['hypersim']:
            TODO  # fov_x and fov_y
        elif componentName in ['blendedMVS']:
            TODO  # intrinsics
        else:
            raise NotImplementedError('Unknown componentName: %s' % componentName)

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()
