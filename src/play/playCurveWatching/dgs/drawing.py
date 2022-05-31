# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from codes_py.toolbox_framework.framework_util_v2 import splitPDSR
from codes_py.np_ext.mat_io_v1 import pSaveMat
from codes_py.py_ext.misc_v1 import mkdir_full


bt = lambda s: s[0].upper() + s[1:]


def drawing(curveTupList, drawingFileName, xlim=None, ylim=None, ylabel=None):
    # set your set
    gauKernelSize = 3
    gauSigma = 1
    curveTupList = curveTupList

    # meta
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
    curveGroupName = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    gauKernel = signal.gaussian(gauKernelSize, gauSigma)
    gauKernel = gauKernel / gauKernel.sum()

    Xrecord = []
    Yrecord = []

    for curveTup in curveTupList:
        curvePDSR, curveLossName, curveNickName, curveColor, curveLinestyle = \
            curveTup
        print('Processing curve for %s' % curveNickName)
        curveP, curveD, curveS, curveR = splitPDSR(curvePDSR)
        lossOrStat = 'loss' if curveLossName.startswith('loss') else 'stat'

        logDir = projRoot + 'v/P/%s/%s/%s/%s/' % (curveP, curveD, curveS, curveR)
        logFileName = logDir + 'trainingLog.txt'
        with open(logFileName, 'r') as f:
            lines = [x.strip().rstrip() for x in f.readlines()
                     if x.strip().startswith('[%s] %s Iter' % (bt(lossOrStat), curveLossName))]

        X = []
        Y = []
        for line in lines:
            ls = line.split(' ')
            assert len(ls) == 4
            assert ls[2].endswith(':')
            X.append(int(ls[2][4:-1]))
            Y.append(float(ls[3]))

        # smoothing
        if len(X) < 5:
            continue
        X = np.array(X, dtype=np.int32)
        Y = np.array(Y, dtype=np.float32)
        ind = np.argsort(X)
        X = X[ind]
        Y = Y[ind]
        Y = np.convolve(Y, gauKernel, mode='valid')
        assert gauKernelSize % 2 == 1
        r = int((gauKernelSize - 1) / 2)
        X = X[r:-r]

        Xrecord.append(X)
        Yrecord.append(Y)

        # plot
        plt.plot(X, Y, c=curveColor, linestyle=curveLinestyle, label=curvePDSR)

    plt.legend()
    if xlim:
        assert len(xlim) == 2
        plt.xlim(left=xlim[0], right=xlim[1])
    if ylim:
        assert len(ylim) == 2
        plt.ylim(bottom=ylim[0], top=ylim[1])

    plt.grid()
    plt.xlabel('# of Iterations')
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('Values (e.g. loss or benchmarking IoUs, See Annotations)')
    saveRoot = projRoot + 'cache/debuggingCenter/playCurveWatching/%s/' % \
               curveGroupName
    mkdir_full(saveRoot)
    plt.savefig(saveRoot + drawingFileName + '.png')
    plt.clf()
    print('Curve Plot Saved!')
    