# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from UDLv2 import udl
from codes_py.np_ext.mat_io_v1 import pSaveMat
from collections import OrderedDict
import pickle
import csv
import scipy
import json
import numpy as np


projRoot = '../../../'


def getSemanticFromRep():
    dataset = 'rep'
    m = udl('_A1_m', dataset)
    houseNameList = udl('_A11_houseNameList', dataset)

    cateRecord = []
    for j in range(m):
        print('Processing Z_semantic for %s: %d / %d' % (dataset, j, m))
        with open(projRoot + 'remote_fastdata/replica/%s/habitat/info_semantic.json' % houseNameList[j], 'r') as f:
            info = json.load(f)
        cateSet = set([x['class_name'] for x in info['objects']])
        cateList = list(cateSet)
        cateRecord += cateList
    cateRecord = list(set(cateRecord))
    return cateRecord


def getSemanticFromMp():
    dataset = 'mp'
    with open(projRoot + 'remote_slowdata2/mp/v1/Matterport/metadata/category_mapping.tsv', 'r') as f:
        tsv = csv.reader(f, delimiter='\t')
        r = [x for x in tsv][1:]
        for k in range(len(r)):
            assert len(r[k]) == 18
            '''
            if r[k][1] != r[k][2]:
                print(r[k])
            '''
            # Conclusion: You should rely on r[k][2] for more accurate
        cateSet = set([x[2] for x in r[1:]])
        cateList = list(cateSet)
    return cateList


def main():
    cateRep = getSemanticFromRep()
    cateMp = getSemanticFromMp()

    # try to find one for each rep category entry
    matchedInds = -np.ones(len(cateRep), dtype=np.int32)
    for kr in range(len(cateRep)):
        for km in range(len(cateMp)):
            if cateRep[kr] == cateMp[km]:
                matchedInds[kr] = km
    del kr
    del km

    augmentFromRep = [cateRep[k] for k in np.where(matchedInds < 0)[0]]
    cate = cateMp + augmentFromRep

    # cateMp has 1146 categories, cateRep has 90.
    # When matching cateRep into cateMp, there are 27 categories that are unmatched.
    # So Totally resulting 1173 categories (1146 + 27).
    # So 0-1145 is just the inds for matterport3d.
    # rematchedInds (len 90) you can use this to extract the 90 categories from cate. i.e. cate[rematchedInds] == cateRep

    rematchedInds = -np.ones(len(cateRep), dtype=np.int32)
    for kr in range(len(cateRep)):
        for kc in range(len(cate)):
            if cateRep[kr] == cate[kc]:
                rematchedInds[kr] = kc
    assert np.all(rematchedInds >= 0).sum()

    indExtractCateRep = rematchedInds
    indExtractCateMp = np.array(range(len(cateMp)), dtype=np.int32)

    cateRepList = cateRep
    cateRepDict = OrderedDict([])
    for j in range(len(cateRepList)):
        cateRepDict[cateRepList[j]] = j
    cateMpList = cateMp
    cateMpDict = OrderedDict([])
    for j in range(len(cateMpList)):
        cateMpDict[cateMpList[j]] = j
    cateList = cate
    cateDict = OrderedDict([])
    for j in range(len(cateList)):
        cateDict[cateList[j]] = j

    with open(projRoot + 'v/Z/semanticRepMp/unionCategory.pkl', 'wb') as f:
        pickle.dump({
            'unionCateList': cateList,
            'unionCateDict': cateDict,
        }, f)
    with open(projRoot + 'v/Z/semanticRepMp/repCategory.pkl', 'wb') as f:
        pickle.dump({
            'datasetCateList': cateRepList,
            'datasetCateDict': cateRepDict,
            'indExtractCategory': indExtractCateRep,
        }, f)
    with open(projRoot + 'v/Z/semanticRepMp/mpCategory.pkl', 'wb') as f:
        pickle.dump({
            'datasetCateList': cateMpList,
            'datasetCateDict': cateMpDict,
            'indExtractCategory': indExtractCateMp,
        }, f)

    '''
    pSaveMat(projRoot + 'v/Z/semanticRepMp/semanticRepMp.mat', {
        'cateRep': cateRep,
        'cateMp': cateMp,
        'cate': cate,
        'indExtractCateRep': indExtractCateRep,
        'indExtractCateMp': indExtractCateMp,
    })
    '''


if __name__ == '__main__':
    main()
