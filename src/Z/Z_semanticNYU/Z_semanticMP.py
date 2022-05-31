# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from UDLv2 import udl
from codes_py.np_ext.mat_io_v1 import pSaveMat
import pickle
import numpy as np
import csv


projRoot = '../../../'
# mp1145List = udl('_Z_mpCategory')['datasetCateList']
# mp1145Dict = udl('_Z_mpCategory')['datasetCateDict']
nyu40List = udl('_Z_semanticNYU')['nyu40List']
nyu40Dict = udl('_Z_semanticNYU')['nyu40Dict']
# mp1145List[0] = 'air'
# mp1145Dict['air'] = 0

mp1145ToNyu40 = -np.ones((1146, ), dtype=np.int32)
with open(projRoot + 'remote_slowdata2/mp/v1/Matterport/metadata/category_mapping.tsv', 'r') as f:
    tsv = csv.reader(f, delimiter='\t')
    r = [x for x in tsv][1:]
    for k in range(len(r)):
        assert len(r[k]) == 18
        if k == 1239:
            r[k][5] = '40'  # missing data

    mp1659List = [x[1] for x in r]
    mp1659ToNyu40 = np.array([int(x[5]) for x in r], dtype=np.int32)

mp1659List = ['air'] + mp1659List
mp1659ToNyu40 = np.concatenate([
    np.array([0], dtype=np.int32),
    mp1659ToNyu40,
], 0)
mp1659Dict = {mp1659List[j]: j for j in range(1660)}
mp1659RetrieveList = {mp1659Dict[key]: key for key in mp1659Dict.keys()}

# Correction (deleted categories)
mp1659ToNyu40[[9, 64, 1481]] = 40

# Final Check
assert mp1659ToNyu40[0] == 0
assert np.all(mp1659ToNyu40[1:] > 0)

pSaveMat(projRoot + 'v/Z/semanticNYU/semanticMP.mat', {
    'mp1659List': mp1659List,
    'mp1659ToNyu40': mp1659ToNyu40,
})

with open(projRoot + 'v/Z/semanticNYU/semanticMP.pkl', 'wb') as f:
    pickle.dump({
        'mp1659Dict': mp1659Dict,
        'mp1659RetrieveList': mp1659RetrieveList,
    }, f)
