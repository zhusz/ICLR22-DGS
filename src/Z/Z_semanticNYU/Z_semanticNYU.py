# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from scipy.io import loadmat
from codes_py.np_ext.mat_io_v1 import pSaveMat
import pickle
import numpy as np


# get nyu40List
projRoot = '../../../'
tmp = loadmat(projRoot + 'external_codes/sscnet/matlab_code/benchmark/NYUclassMapping40.mat')['className']
nyu40List = [str(tmp[0][j][0]) for j in range(40)]
nyu40List = ['air'] + nyu40List
nyu40Dict = {nyu40List[j]: j for j in range(41)}
nyu40RetrieveList = {nyu40Dict[key]: key for key in nyu40Dict}
del tmp

# get nyu36List
tmp = loadmat(projRoot + 'external_codes/sscnet/matlab_code/ClassMapping.mat')['p5d36class']
nyu36List = [str(tmp[0][j][0]) for j in range(36)]
nyu36List = ['air'] + nyu36List
nyu36Dict = {nyu36List[j]: j for j in range(37)}
nyu36RetrieveList = {nyu36Dict[key]: key for key in nyu36Dict}
del tmp

# get nyu11List
tmp = loadmat(projRoot + 'external_codes/sscnet/matlab_code/ClassMapping.mat')['elevenClass']
nyu11List = [str(tmp[0][j][0]) for j in range(11)]
nyu11List = ['air'] + nyu11List
nyu11Dict = {nyu11List[j]: j for j in range(12)}
nyu11RetrieveList = {nyu11Dict[key]: key for key in nyu11Dict}
del tmp

# get nyu37List. Note the so-called nyu37 is just nyu40! But you need to map the name.
tmp = loadmat(projRoot + 'external_codes/sunrgbd/SUNRGBDtoolbox/Metadata/seg37list.mat')['seg37list']
nyu37List = [str(tmp[0][j][0]) for j in range(37)]
nyu37List = ['air'] + nyu37List
nyu37NameMapping = {nyu37List[j]: nyu40List[j] for j in range(38)}
del tmp

# get nyu40ToNyu36: Note we always map 0 to 0. So a[1] refers to the mapping of the first class.
tmp = loadmat(projRoot + 'external_codes/sscnet/matlab_code/ClassMapping.mat')['mapNYU40to36']
nyu40ToNyu36List = [str(tmp[0][j][0]) for j in range(40)]
nyu40ToNyu36List = ['air'] + nyu40ToNyu36List
nyu40ToNyu36 = np.array([nyu36Dict[nyu40ToNyu36List[j]] for j in range(41)])

# get nyu36ToNyu11: the same as above
tmp = loadmat(projRoot + 'external_codes/sscnet/matlab_code/ClassMapping.mat')['map36to11']
nyu36ToNyu11List = [str(tmp[0][j][0]) for j in range(36)]
nyu36ToNyu11List = ['air'] + nyu36ToNyu11List
nyu36ToNyu11 = np.array([nyu11Dict[nyu36ToNyu11List[j]] for j in range(37)])

pSaveMat(projRoot + 'v/Z/semanticNYU/semanticNYU.mat', {
    'nyu40List': nyu40List,
    # 'nyu40Dict': nyu40Dict,
    'nyu36List': nyu36List,
    # 'nyu36Dict': nyu36Dict,
    'nyu11List': nyu11List,
    # 'nyu11Dict': nyu11Dict,
    'nyu40ToNyu36': nyu40ToNyu36,
    'nyu36ToNyu11': nyu36ToNyu11,
})

with open(projRoot + 'v/Z/semanticNYU/semanticNYU.pkl', 'wb') as f:
    pickle.dump({
        'nyu40Dict': nyu40Dict,
        'nyu40RetrieveList': nyu40RetrieveList,
        'nyu36Dict': nyu36Dict,
        'nyu36RetrieveList': nyu36RetrieveList,
        'nyu11Dict': nyu11Dict,
        'nyu11RetrieveList': nyu11RetrieveList,
        'nyu37NameMapping': nyu37NameMapping,
    }, f)

