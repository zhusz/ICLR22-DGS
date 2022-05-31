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
from UDLv3 import udl
import pickle
import numpy as np
from collections import OrderedDict


def get_shapenetcatesha_to_shapenetcatename():
    with open('./shapenet_categories.txt', 'r') as f:
        lines = f.readlines()
    sha2name = OrderedDict([])
    for line in lines:
        ss = line.strip().rstrip().split(' ')
        ss = [s.strip().rstrip() for s in ss]
        for s in ss:
            t = s.split('&')
            assert len(t) == 2
            assert len(t[0]) == 8
            sha2name[t[0]] = t[1]

    return sha2name


def main():
    sha2name = get_shapenetcatesha_to_shapenetcatename()  # 55 categories from here
    # Note openrooms only contain s 35 of the 55, but we maintain 55 in here.
    sha2name['ceiling_lamp'] = 'ceiling_lamp'
    sha2name['curtain'] = 'curtain'
    sha2name['door'] = 'door'
    sha2name['window'] = 'window'
    sha2name['layout'] = 'layout'

    openrooms60List = ['air'] + list(sha2name.values())
    openrooms60Dict = OrderedDict([])
    openrooms60Dict['air'] = 0
    openrooms60RetrieveList = OrderedDict([])
    openrooms60RetrieveList[0] = 'air'
    for i, name in enumerate(sha2name.values()):
        openrooms60Dict[name] = i + 1
        openrooms60RetrieveList[i + 1] = name

    # label the semanitc mapping to later
    with open(projRoot + 'v/Z/semanticOpenrooms/semanticOpenrooms.pkl', 'wb') as f:
        pickle.dump({
            'openrooms60List': openrooms60List,
            'openrooms60Dict': openrooms60Dict,
            'openrooms60RetrieveList': openrooms60RetrieveList,
            'openroomsSha2name': sha2name,
        }, f)

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()
