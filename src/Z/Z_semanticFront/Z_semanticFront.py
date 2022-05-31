# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# any env
import os
import csv
import numpy as np
import pickle


def main():
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'

    # self super category
    _SUPER_CATEGORIES_3D = [
        {'id': 1, 'category': 'Cabinet/Shelf/Desk'},
        {'id': 2, 'category': 'Bed'},
        {'id': 3, 'category': 'Chair'},
        {'id': 4, 'category': 'Table'},
        {'id': 5, 'category': 'Sofa'},
        {'id': 6, 'category': 'Pier/Stool'},
        {'id': 7, 'category': 'Lighting'},
        {'id': 8, 'category': 'Other'},
    ]
    frontSuper8List = ['air'] + [x['category'] for x in _SUPER_CATEGORIES_3D]
    frontSuper8Dict = {frontSuper8List[c]: c for c in range(len(frontSuper8List))}

    # self category
    count = 0
    front77List = ['air']
    with open(projRoot + 'external_codes/BlenderProcOld/resources/'
                         'front_3D/3D_front_mapping.csv', 'r') as f:
        rows = csv.reader(f)
        for row in rows:
            if row[0] == 'id':
                continue
            if row[1] not in ['void', 'hole']:
                front77List.append(row[1])
                count += 1
    front77Dict = {front77List[c]: c for c in range(78)}

    # relation with nyu40
    front77ToNyu40 = -np.ones((78, ), dtype=np.int32)
    front77ToNyu40[0] = 0
    with open(projRoot + 'external_codes/BlenderProcOld/resources/'
                         'front_3D/3D_front_nyu_mapping.csv', 'r') as f:
        rows = csv.reader(f)
        for row in rows:
            if row[0] == 'id':
                continue
            nyu40cateID = int(row[0])
            front77cateName = row[1]
            if front77cateName in ['void', 'hole']:
                continue
            front77cateID = int(front77Dict[front77cateName])
            assert 0 <= front77cateID <= 77
            front77ToNyu40[front77cateID] = nyu40cateID

    with open(projRoot + 'v/Z/semanticFront/semanticFront.pkl', 'wb') as f:
        pickle.dump({
            'frontSuper8List': frontSuper8List,
            'frontSuper8Dict': frontSuper8Dict,
            'front77List': front77List,
            'front77Dict': front77Dict,
            'front77ToNyu40': front77ToNyu40,
        }, f)


if __name__ == '__main__':
    main()
