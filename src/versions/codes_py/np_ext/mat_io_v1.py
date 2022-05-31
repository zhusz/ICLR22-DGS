from scipy.io import savemat, loadmat
from ..py_ext.misc_v1 import liststr2cellstr, cellstr2liststr
import numpy as np
import copy


def pSaveMat(matFileName, x, m=None):
    assert type(matFileName) is str
    assert type(x) is dict

    y = copy.deepcopy(x)
    for k in y.keys():
        if type(y[k]) is np.ndarray and len(y[k].shape) == 1:
            if m is not None:
                assert len(y[k]) == m
            y[k] = y[k][:, None]
        if type(y[k]) is list and type(y[k][0]) is str:
            if m is not None:
                assert len(y[k]) == m
            for x in y[k]:
                assert type(x) is str
            y[k] = liststr2cellstr(y[k])

    savemat(matFileName, y)


def pLoadMat(matFileName):
    mat = loadmat(matFileName)
    pProcessMat(mat)
    return mat


def pProcessMat(mat):  # non functional!
    for k in ['__header__', '__version__', '__globals__']:
        if k in list(mat.keys()):
            del mat[k]
    for k in mat.keys():  # special process at one time
        if not k.startswith('__'):
            if 'object' in str(mat[k].dtype):  # str
                # Suppose to be the relative file path
                # Otherwise, do not save str into the mat file
                if k.endswith('FileList'):  # filepath str, adding v_root
                    mat[k] = [v_root + x for x in cellstr2liststr(mat[k])]
                else:  # common str
                    mat[k] = cellstr2liststr(mat[k])
            elif len(mat[k].shape) == 2 and mat[k].shape[1] == 1:  # squeeze related
                if mat[k].shape[0] == 1:  # scalar
                    mat[k] = int(mat[k]) if 'int' in str(mat[k].dtype) else float(mat[k])
                else:  # vector
                    mat[k] = mat[k].squeeze()
        else:
            raise KeyError('Your mat should not include a key startswith __: keyName: %s' % k)



