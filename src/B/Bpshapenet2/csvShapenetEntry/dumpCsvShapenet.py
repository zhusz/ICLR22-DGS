# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
sys.path.append(projRoot + 'src/versions/')
from configs_registration import getConfigGlobal
from ..testDataEntry.testDataEntryPool import getTestDataEntryDict
from ..approachEntryPool import getApproachEntryDict
from codes_py.py_ext.misc_v1 import mkdir_full
from codes_py.toolbox_framework.framework_util_v2 import splitPDSRI, splitPDDrandom
from codes_py.py_ext.misc_v1 import mkdir_full, tabularPrintingConstructing, tabularCvsDumping
from codes_py.toolbox_3D.mesh_v1 import vertInfo2faceVertInfoNP
from collections import OrderedDict
import pickle
import numpy as np


bt = lambda s: s[0].upper() + s[1:]
governedScriptTagList = ['1D', '1D128']


def _wrong():
    print('Something is wrong. Please check!')
    import ipdb
    ipdb.set_trace()
    print(1 + 1)


def assertBenchmarkings(bsv0_toStore, bsv0_loaded, **kwargs):
    raise NotImplementedError('TODO')
    return


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


def get_s(bsv0_retrieveList_dict, approachEntryDict, testDataEntry, **kwargs):
    # bsv0_retrieveList_dict[approachNickName][index] is bsv0

    # from testDataEntry
    indChosen = testDataEntry['indChosen']

    s = OrderedDict([])
    allCatName = list(getCats()[0].keys())
    for approachNickName in bsv0_retrieveList_dict.keys():
        approachShownName = approachEntryDict[approachNickName]['approachShownName']
        s[approachNickName] = OrderedDict([])
        s[approachNickName]['method'] = approachShownName

        catRecord = OrderedDict([])
        for catName in allCatName:
            catRecord[catName] = []

        # benPool = ['corenetCubeIou']

        for j in indChosen:
            bsv0 = bsv0_retrieveList_dict[approachNickName][j]
            catName = bsv0['catName']
            catRecord[catName].append(bsv0)
        record_for_mean = []
        for catName in allCatName:
            if len(catRecord[catName]) == 0:
                s[approachNickName][catName + 'Iou'] = '0'
            else:
                s[approachNickName][catName + 'Iou'] = \
                    '%.01f' % \
                    (100 *
                     (sum([ben['finalBenCorenetCubeIou'] for ben in catRecord[catName]]) / len(catRecord[catName])))
                record_for_mean.append(sum([ben['finalBenCorenetCubeIou'] for ben in catRecord[catName]]) / len(catRecord[catName]))
        s[approachNickName]['meanIou'] = '%.01f' % (100 * sum(record_for_mean) / len(record_for_mean))
    return s

