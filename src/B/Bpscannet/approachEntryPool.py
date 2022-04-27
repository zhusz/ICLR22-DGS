# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def getApproachEntryDict():
    approachEntryList = [  # (approachNickName, methodologyName (PDSRI), scriptTag, approachShownName)
        # approachNickname is the primary key now
        ('DGS-New', 'PscannetD0dgsS101RnewIlatest', '1E', None),
        ('DGS-Submission-Time', 'PscannetD0dgsS101RreleaseIlatest', '1Epp1', None),
        ('DGS-Submission-Time-No-PostProcess', 'PscannetD0dgsS101RreleaseIlatest', '1E', None),
    ]

    approachEntryDict = {approachEntry[0]: {
        'approachNickName': approachEntry[0],
        'methodologyName': approachEntry[1],
        'scriptTag': approachEntry[2],
        'approachShownName': approachEntry[3] if approachEntry[3] else approachEntry[0],
    } for approachEntry in approachEntryList}

    return approachEntryDict
