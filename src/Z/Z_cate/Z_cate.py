# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle


projRoot = '../../../'

cateListPix3d = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
cateListPix3d = {c: cateListPix3d[c] for c in range(len(cateListPix3d))}
cateDictPix3d = {cateListPix3d[k]: k for k in range(len(cateListPix3d))}

cateListSunrgbdV1 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling', 'books', 'fridge', 'tv', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag']
cateListSunrgbdV1 = {c: cateListSunrgbdV1[c] for c in range(len(cateListSunrgbdV1))}
cateDictSunrgbdV1 = {cateListSunrgbdV1[k]: k for k in range(len(cateListSunrgbdV1))}

with open(projRoot + 'v/Z/cate/cate.pkl', 'wb') as f:
    pickle.dump({
        'cateListPix3d': cateListPix3d,
        'cateDictPix3d': cateDictPix3d,
        'cateListSunrgbdV1': cateListSunrgbdV1,
        'cateDictSunrgbdV1': cateDictSunrgbdV1,
    }, f)

# -1 - cross-dataset-undefined
# -2 - uncounted  (typical in sunrgbdV1)
