# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def getApproachEntryDict():
    approachEntryList = [  # (approachNickName, methodologyName (PDSRI), scriptTag, approachShownName)  # (first1%, full)
        # approachNickname is the primary key now
        ('Oracle-CoReNet-64', 'Pshapenet2D0corenetS0RreleaseIlatest', '1D', None),
        ('Oracle-CoReNet-128', 'Pshapenet2D0corenetS0RreleaseIlatest', '1D128', None),

        ('Ours-DGS-64', 'Pshapenet2D0corenetdgsS2RreleaseIlatest', '1D', None),
        ('Ours-DGS-128', 'Pshapenet2D0corenetdgsS2RreleaseIlatest', '1D128', None),

        ('Oracle-DISN-DVR-64', 'Pshapenet2D0disndvrS2RreleaseI500000', '1D', None),
        ('Oracle-DISN-DVR-128', 'Pshapenet2D0disndvrS2RreleaseI500000', '1D128', None),

        ('Ours-DGS-Best-64', 'Pshapenet2D0disndvrdgsS2RreleaseIlatest', '1D', None),
        ('Ours-DGS-Best-128', 'Pshapenet2D0disndvrdgsS2RreleaseIlatest', '1D128', None),
    ]

    approachEntryDict = {approachEntry[0]: {
        'approachNickName': approachEntry[0],
        'methodologyName': approachEntry[1],
        'scriptTag': approachEntry[2],
        'approachShownName': approachEntry[3] if approachEntry[3] else approachEntry[0],
    } for approachEntry in approachEntryList}

    return approachEntryDict
