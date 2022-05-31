# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from UDLv3 import udl
from codes_py.np_ext.mat_io_v1 import pSaveMat
import numpy as np
import pickle


projRoot = '../../../'

nyu40RetrieveList = udl('_Z_semanticNYU')['nyu40RetrieveList']

pix3d10List = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe', 'structure']
pix3d10List = ['air'] + pix3d10List
pix3d10Dict = {pix3d10List[j]: j for j in range(11)}
pix3d10RetrieveList = {j: pix3d10List[j] for j in range(11)}

nyu40ToPix3d10Dict = {  # borrowed from the total3dUnderstand paper
    'air': 0,
    'wall': 10,  # zhuzhu
    'floor': 10,  # zhuzhu
    'cabinet': 8,
    'bed': 1,
    'chair': 3,
    'sofa': 5,
    'table': 6,
    'door': 8,
    'window': 9,
    'bookshelf': 2,
    'picture': 9,
    'counter': 9,
    'blinds': 9,
    'desk': 4,
    'shelves': 2,
    'curtain': 9,
    'dresser': 8,
    'pillow': 9,
    'mirror': 9,
    'floor mat': 9,
    'clothes': 9,
    'ceiling': 10,  # zhuzhu
    'books': 9,
    'refridgerator': 8,
    'television': 8,
    'paper': 9,
    'towel': 9,
    'shower curtain': 9,
    'box': 8,
    'whiteboard': 8,
    'person': 9,
    'night stand': 8,
    'toilet': 9,
    'sink': 9,
    'lamp': 9,
    'bathtub': 9,
    'bag': 8,
    'otherstructure': 10,  # zhuzhu
    'otherfurniture': 8,  # zhuzhu
    'otherprop': 9,  # zhuzhu
}

nyu40ToPix3d10 = np.array([nyu40ToPix3d10Dict[nyu40RetrieveList[j]] for j in range(41)], dtype=np.int32)

pSaveMat(projRoot + 'v/Z/semanticNYU/semanticPIX3D.mat', {
    'pix3d10List': pix3d10List,
    'nyu40ToPix3d10': nyu40ToPix3d10,
})

with open(projRoot + 'v/Z/semanticNYU/semanticPIX3D.pkl', 'wb') as f:
    pickle.dump({
        'pix3d10Dict': pix3d10Dict,
        'pix3d10RetrieveList': pix3d10RetrieveList,
    }, f)
