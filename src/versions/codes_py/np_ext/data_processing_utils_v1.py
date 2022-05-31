import numpy as np
from matplotlib.cm import get_cmap


def determinedArbitraryPermutation(n):
    ind = np.array(range(n)).tolist()
    ind = ind[::2] + ind[1::2]
    ind = ind[::3] + ind[1::3] + ind[2::3]
    ind = ind[::5] + ind[1::5] + ind[2::5] + ind[3::5] + ind[4::5]
    return ind


def determinedArbitraryPermutedVector(a):
    assert len(a.shape) == 1
    ind = determinedArbitraryPermutation(a.shape[0])
    return a[ind]


# Please do not call this function again
def determinedArbitraryPermutedVector2(a):
    assert len(a.shape) == 1
    ind = determinedArbitraryPermutation(a.shape[0])
    a[ind] = a
    return a


# This is still wrong. Please stop using this function. Now use d0-d3
def determinedArbitraryPermutedVector2correct(a):
    assert len(a.shape) == 1
    b = a.copy()
    ind = determinedArbitraryPermutation(a.shape[0])
    b[ind] = b
    return b


def determinedArbitraryPermutedVector2correctcorrect(a):
    assert len(a.shape) == 1
    b = a.copy()
    ind = determinedArbitraryPermutation(a.shape[0])
    b[ind] = a
    return b


def d0(n, k):  # generalized version of determinedArbitraryPermutation, with a different ordering
    prime_list = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
        47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    assert k >= 2
    assert type(k) is int
    ind = np.array(range(n)).tolist()
    for p in prime_list:
        tmp = []
        order = list(range(p))
        order = order[1::2] + order[0::2]  # [::-1]
        order = order[1::3] + order[0::3] + order[2::3]  # [::-1] + order[2::3]
        for t in order:
            if t % 2 == 0:
                tmp += ind[t::p]  # [::-1]
            else:
                tmp += ind[t::p]
        ind = tmp
        if p > k:
            break
    return ind


def d1(a, k):
    assert len(a.shape) == 1
    ind = d0(a.shape[0], k)
    return a[ind]


def d2(a, k):
    assert len(a.shape) == 1
    b = a.copy()
    ind = d0(a.shape[0], k)
    b[ind] = a
    return b


def uniqueFloatRearrange(a, ifColorShuffle=False):
    # input a can be any shape, but must be 'int' dtype
    # output b is in the same shape as a, dtype float32, but the range in [0, 1), with the largest val to be 1. - 1. / Lu


    # Example Input: a is the objID, so it might be not the full 0~n-1, but distinct number represents different ID
    # Example Output: the output can be directly fed into your cmap.

    assert 'int' in str(a.dtype)
    u = np.unique(a)
    Lu = len(u)

    # For permutation without randomness
    if ifColorShuffle:
        ind = determinedArbitraryPermutation(Lu)
        u = u[ind]

    b = -np.ones(a.shape, dtype=np.float32)
    for k in range(Lu):
        b[a == u[k]] = float(k) / Lu
    return b


def dictGrouping(d, numGroup, ifShuffle):
    # put all the values of the dict d into groups
    # if m % numGroup != 0, then discard the last remaining

    m = None
    for k in d.keys():
        assert type(d[k]) is np.ndarray
        if m is None:
            m = d[k].shape[0]
        else:
            assert m == d[k].shape[0]

    numSamplePerGroup = int(float(m) / numGroup)
    if ifShuffle:
        shuffleRa = np.random.permutation(numGroup * numSamplePerGroup)
    else:
        shuffleRa = np.arange(numGroup * numSamplePerGroup, dtype=np.int32)
    dOut = [{key: d[key][shuffleRa[g * numSamplePerGroup:(g + 1) * numSamplePerGroup]] for key in d.keys()}
            for g in range(numGroup)]

    return dOut


