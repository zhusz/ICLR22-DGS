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


projRoot = '../../../'
datasetCateList_rep_0_89 = udl('_Z_repCategory')['datasetCateList']
datasetCateDict_rep_0_89 = udl('_Z_repCategory')['datasetCateDict']
nyu40List = udl('_Z_semanticNYU')['nyu40List']

datasetCateList_rep_0_89_to_nyu40 = -np.ones((len(datasetCateList_rep_0_89), ), dtype=np.int32)
for j in range(len(datasetCateList_rep_0_89)):
    match = np.array([nyu40List[i] in datasetCateList_rep_0_89[j] for i in range(41)], dtype=np.int32)

    # special class name handling
    match[0] = 0  # the air class should not be matched
    if datasetCateList_rep_0_89[j] == 'wall-cabinet':
        match[1] = 0  # choose cabinet over wall

    if not match.sum() <= 1:
        import ipdb
        ipdb.set_trace()
    if match.sum() == 1:
        datasetCateList_rep_0_89_to_nyu40[j] = np.argmax(match)

    # special class name handling
    if datasetCateList_rep_0_89[j] == 'beanbag':
        datasetCateList_rep_0_89_to_nyu40[j] = 6  # sofa rather than bag
    if datasetCateList_rep_0_89[j] == 'indoor-plant':
        datasetCateList_rep_0_89_to_nyu40[j] = 40  # otherprop rather than door
    if datasetCateList_rep_0_89[j] == 'desk-organizer':
        datasetCateList_rep_0_89_to_nyu40[j] = 40  # otherprop rather than door

# Check existing match
# for j in range(len(datasetCateList_rep_0_89)):
#     if datasetCateList_rep_0_89_to_nyu40[j] != -1:
#         print('%20s: %s' % (datasetCateList_rep_0_89[j], nyu40List[datasetCateList_rep_0_89_to_nyu40[j]]))

# Print the remaining
# print([datasetCateList_rep_0_89[j] for j in range(90) if datasetCateList_rep_0_89_to_nyu40[j] == -1])

# Assign the remaining
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['rug']] = 2  # floor
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['refrigerator']] = 24  # refrigerator. NYU's spelling is wrongdatasetCateDict_rep_0_89[
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['bench']] = 6  # sofa
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['comforter']] = 4  # bed
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['blanket']] = 4  # bed
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['nightstand']] = 32  # night stand
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['stool']] = 5  # chair
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['shelf']] = 15  # shelves
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['vent']] = 1  # wall
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['mat']] = 2  # floor
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['rack']] = 39  # otherfurniture
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['stair']] = 38  # otherstructure
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['pillar']] = 38  # othersturcture
datasetCateList_rep_0_89_to_nyu40[datasetCateDict_rep_0_89['book']] = 23  # books
assignToOtherprop = ['basket', 'plant-stand', 'switch', 'anonymize_text', 'tv-screen', 'faucet', 'candle', 'pipe', 'remote-control', 'scarf', 'bowl', 'bin', 'shoe', 'knife-block', 'undefined', 'vase', 'scuplture', 'utensil-holder', 'coaster', 'camera', 'bottle', 'monitor', 'bike', 'cloth', 'pot', 'cooktop', 'panel', 'plate', 'cup', 'non-plane', 'clothing', 'chopping-board', 'major-appliance', 'clock', 'kitchen-utensil', 'pan', 'shower-stall', 'handrail', 'sculpture', 'other-leaf', 'cushion', 'umbrella', 'tv-stand', 'small-appliance']
for j in range(90):
    # Assign to 'otherprop' 40
    if datasetCateList_rep_0_89[j] in assignToOtherprop:
        assert datasetCateList_rep_0_89_to_nyu40[j] == -1
        datasetCateList_rep_0_89_to_nyu40[j] = 40
print([datasetCateList_rep_0_89[j] for j in range(90) if datasetCateList_rep_0_89_to_nyu40[j] == -1])

# Final check
assert np.all(datasetCateList_rep_0_89_to_nyu40 > 0)

# Check existing match
for j in range(len(datasetCateList_rep_0_89)):
    if datasetCateList_rep_0_89_to_nyu40[j] != -1:
        print('%20s: %s' % (datasetCateList_rep_0_89[j], nyu40List[datasetCateList_rep_0_89_to_nyu40[j]]))

# update into rep91List
rep91List = ['air'] + datasetCateList_rep_0_89
rep91Dict = {rep91List[j]: j for j in range(91)}
rep91RetrieveList = {rep91Dict[key]: key for key in rep91Dict.keys()}
rep91ToNyu40 = np.concatenate([
    np.array([0], dtype=np.int32),
    datasetCateList_rep_0_89_to_nyu40,
], 0)

pSaveMat(projRoot + 'v/Z/semanticNYU/semanticREP.mat', {
    'rep91List': rep91List,
    'rep91ToNyu40': rep91ToNyu40,
})

with open(projRoot + 'v/Z/semanticNYU/semanticREP.pkl', 'wb') as f:
    pickle.dump({
        'rep91Dict': rep91Dict,
        'rep91RetrieveList': rep91RetrieveList,
    }, f)
