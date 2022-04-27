# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


def addToSummary0Txt0BrInds(
        summary0, txt0, brInds, approachEntryDict, bsv0_forVis_dict, **kwargs):
    # Not Functional
    for approachNickName in approachEntryDict.keys():
        approachEntry = approachEntryDict[approachNickName]
        approachShownName = approachEntry['approachShownName']
        bsv0_forVis = bsv0_forVis_dict[approachNickName]

        summary0['img_%s' % approachShownName] = bsv0_forVis['img'].transpose((1, 2, 0))
        txt0.append('')
        brInds.append(len(summary0))

    return summary0, txt0, brInds


def main():
    # set your set
    testDataNickName = os.environ['DS']

    # general
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
    testDataEntryDict = getTestDataEntryDict(wishedTestDataNickName=[testDataNickName])
    approachEntryDict = getApproachEntryDict()
    Btag = os.path.realpath(__file__).split('/')[-3]
    assert Btag.startswith('B')
    Btag_root = projRoot + 'v/B/%s/' % Btag
    assert os.path.isdir(Btag_root)
    testDataEntry = testDataEntryDict[testDataNickName]

    # html setups
    visualDir = projRoot + 'cache/B_%s/%s_dumpHtmlForPrepick/' % (Btag, testDataNickName)
    mkdir_full(visualDir)
    htmlStepper = HTMLStepper(visualDir, 100, testDataNickName)
    dumpDir = visualDir
    os.makedirs(dumpDir, exist_ok=True)

