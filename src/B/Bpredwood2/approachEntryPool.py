# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def getApproachEntryDict():
    approachEntryList = [  # (approachNickName, methodologyName (PDSRI), scriptTag, approachShownName)
        # approachNickname is the primary key now

        # ('PredwoodA30D0dgsShybridNeMajor256R0I20WOn1E', 'PredwoodA30D0dgsShybridNeMajor256R0I200002', '1E', None),
    ]

    approachEntryDict = {approachEntry[0]: {
        'approachNickName': approachEntry[0],
        'methodologyName': approachEntry[1],
        'scriptTag': approachEntry[2],
        'approachShownName': approachEntry[3] if approachEntry[3] else approachEntry[0],
    } for approachEntry in approachEntryList}

    return approachEntryDict
