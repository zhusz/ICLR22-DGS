# (tfconda)
import numpy as np

def id2mat(idMat, verbose=None):  # functional
    # Input: idMat: m * nOrders. (e.g. for a clip-frame order-2 video organizing, nOrders==2)
    # Output: orderTree  (python dict recursive)

    orderTree = {}
    assert(len(idMat.shape) == 2)
    m = idMat.shape[0]
    nOrders = idMat.shape[1]
    assert 'int' in str(idMat.dtype)

    for j in range(m):
        if verbose is not None and j % verbose == 0:
            print('    processing id2map (nOrders = %d): %d / %d' % (nOrders, j, m))

        nodePointer = orderTree
        for o in range(nOrders - 1):  # the last level should not be a {}, and should be done in a separate way below
            orderLevelOIndex = int(idMat[j, o])
            if orderLevelOIndex not in list(nodePointer.keys()):
                nodePointer[orderLevelOIndex] = {}
            nodePointer = nodePointer[orderLevelOIndex]

        # the last level
        orderLevelLastIndex = int(idMat[j, nOrders - 1])
        if orderLevelLastIndex in list(nodePointer.keys()):
            a = nodePointer[orderLevelLastIndex]
            idA = idMat[a]
            idJ = idMat[j]
            raise ValueError('Conflict of ids: \nSample %d: %s\nSample %d: %s' % (a, str(idA), j, str(idJ)))

        nodePointer[orderLevelLastIndex] = j

    return orderTree
