# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
from UDLv3 import udl
from collections import OrderedDict
import numpy as np
import json
import pickle


def getCats():
    cats = {
        "watercraft": "04530566",
        "rifle": "04090263",
        "display": "03211117",
        "lamp": "03636649",
        "speaker": "03691459",
        "cabinet": "02933112",
        "chair": "03001627",
        "bench": "02828884",
        "car": "02958343",
        "airplane": "02691156",
        "sofa": "04256520",
        "table": "04379243",
        "phone": "04401088",
    }
    cats = OrderedDict(cats)
    cats_inverse = {cats[k]: k for k in cats.keys()}

    cat_ids = list(cats.values())
    cat_ids_to_catOrder = OrderedDict([
        (cat_ids[c], c)
        for c in range(len(cat_ids))
    ])
    return cats, cats_inverse, cat_ids, cat_ids_to_catOrder


def main():
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'

    assert dataset == 'corenetChoySingleRendering'
    A1_full = udl('pkl_A1b_', 'corenetSingleRendering')
    m_full = A1_full['m']
    catIDList_full = A1_full['catIDList']
    del A1_full

    cats, cats_inverse, cat_ids, cat_ids_to_catOrder = getCats()
    flagIsChoy = np.zeros((m_full, ), dtype=bool)
    for j_full, catID_full in enumerate(catIDList_full):
        if catID_full in cat_ids:
            flagIsChoy[j_full] = 1

    # make up the new  # do the remaining
    extract = lambda l, flag: [x for j, x in enumerate(l) if flag[j]]
    m = int(flagIsChoy.sum())
    A1_full = udl('pkl_A1b_', 'corenetSingleRendering')
    corenetShaTagList = extract(A1_full['corenetShaTagList'], flagIsChoy)
    corenetShaTagDict = {}
    for index, corenetShaTag in enumerate(corenetShaTagList):
        corenetShaTagDict[corenetShaTag] = index
    catIDList = extract(A1_full['catIDList'], flagIsChoy)
    shaIDList = extract(A1_full['shaIDList'], flagIsChoy)
    catNameList = [cats_inverse[catID] for catID in catIDList]
    d = {
        'm': m,
        'flagSplit': A1_full['flagSplit'][flagIsChoy],
        'sptTagList': extract(A1_full['sptTagList'], flagIsChoy),
        'splitTagList': extract(A1_full['splitTagList'], flagIsChoy),
        'corenetShaH3TagList': extract(A1_full['corenetShaH3TagList'], flagIsChoy),
        'corenetShaTagList': corenetShaTagList,
        'corenetShaTagDict': corenetShaTagDict,

        'camObj2World': A1_full['camObj2World'][flagIsChoy],
        'camWorld2CamTheirs': A1_full['camWorld2CamTheirs'][flagIsChoy],
        'cameraTransform': A1_full['cameraTransform'][flagIsChoy],
        'camRObj2CamOurs': A1_full['camRObj2CamOurs'][flagIsChoy],
        'camTObj2CamOurs': A1_full['camTObj2CamOurs'][flagIsChoy],

        'winWidth': A1_full['winWidth'][flagIsChoy],
        'winHeight': A1_full['winHeight'][flagIsChoy],
        'focalLengthWidth': A1_full['focalLengthWidth'][flagIsChoy],
        'focalLengthHeight': A1_full['focalLengthHeight'][flagIsChoy],
        'f0': A1_full['f0'],

        'catIDList': catIDList,
        'shaIDList': shaIDList,
        'catNameList': catNameList,

        'indexInFull': np.where(flagIsChoy)[0],
    }
    for k in d.keys():
        if type(d[k]) not in [int, float]:
            print('%s %s' % (k, len(d[k])))
    with open(projRoot + 'v/A/%s/A1b_order1Supp.pkl' % dataset, 'wb') as f:
        pickle.dump(d, f)


if __name__ == '__main__':
    main()