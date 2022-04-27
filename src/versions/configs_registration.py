# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def getConfigGlobal(P, D, S, R, wishedClassNameList=None):
    returnExportedClasses = __import__(
        '%s.%s.extension' % (P, D), globals(), locals(), ['returnedExportedClasses']
    ).returnExportedClasses
    exportedClasses = returnExportedClasses(wishedClassNameList)
    if S == 'SNono':
        getConfigFunc = lambda P, D, S, R: None
    else:
        getConfigFunc = __import__(
            '%s.%s.config_%s' % (P, D, S), globals(), locals(), ['getConfigFunc']
        ).getConfigFunc

    return {
        'getConfigFunc': getConfigFunc,
        'exportedClasses': exportedClasses,
    }
