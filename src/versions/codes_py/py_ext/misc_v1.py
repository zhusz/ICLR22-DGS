from collections import OrderedDict
from prettytable import PrettyTable
import csv
import copy
import numpy as np
import os

# ----------------------- Loading Mat to python --------------------- #
def __list_take_off_dim(x, num_of_dim):  # nonfunctional, banning being called externally
    out = []
    for i in range(len(x)):
        y = x[i]
        for j in range(num_of_dim):
            y = y[0]
        out.append(y)
    return out

def cellstr2liststr(cellstr):  # functional
    return copy.deepcopy([str(x) for x in __list_take_off_dim(cellstr, 2)])

def liststr2cellstr(liststr):  # functional
    assert type(liststr) is list
    cellstr = np.zeros((len(liststr),), dtype=object)
    for j in range(len(liststr)):
        assert type(liststr[j]) is str
        cellstr[j] = liststr[j]
    return copy.deepcopy(np.expand_dims(cellstr, 1))

# --------------------- Experimental platform ---------------------- #

def get_iter_from_filestring(filestring):  # functional  # *-iter.*
    ida = str.rfind(filestring, '-')
    idb = str.rfind(filestring, '.')
    assert(ida > 0)
    assert(idb > 0)
    return int(filestring[ida+1:idb])

# --------------------- File System ----------------------- #
def mkdir_full(path):  # functional
    if os.path.isdir(path): return
    if path[-1] == '/': path = path[:-1]
    sl = path.rfind('/')
    if not os.path.isdir(path[:sl]):
        mkdir_full(path[:sl])
    os.mkdir(path)

def get_latest_model_iter(model_save_root):  # functional
    fn = os.listdir(model_save_root)
    if not fn:
        return 0
    iter_max = -1
    for i in range(len(fn)):
        if not fn[i][-3:] == 'pth':
            continue
        iter_now = get_iter_from_filestring(fn[i])
        if iter_now > iter_max:
            iter_max = iter_now
    return iter_max

# ---------------------- list operations -------------------------- #
def strictly_increasing(L):  # functional
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):  # functional
    return all(x>y for x, y in zip(L, L[1:]))

def non_increasing(L):  # functional
    return all(x>=y for x, y in zip(L, L[1:]))

def non_decreasing(L):  # functional
    return all(x<=y for x, y in zip(L, L[1:]))

'''  This is non-functional, so just being here for your reference.
def get_first_true(L, fn):
    return next(x[0] for x in enumerate(L) if fn(x[1]))
'''

# -------------------- Data Format Transform (Python Basic Types) -------------------- #
def retrieveList2List(retrieveList, ifCheck, valueOfMissingKey=None, tolerantMaxKeyInt=10000):
    keys = list(retrieveList.keys())
    if ifCheck:
        for key in keys:
            if type(key) is not int or key < 0 or key >= tolerantMaxKeyInt:
                print('Invalid key from retrieveList. Please check!')
                import ipdb
                ipdb.set_trace()

    keys = sorted(keys)
    now = 0
    outputList = []
    for key in keys:
        while now < key:
            outputList.append(valueOfMissingKey)
            now += 1
        assert key == now
        outputList.append(retrieveList[key])
        now += 1
    return outputList


# -------------------- tabular printing / csv dumping --------------------------- #
def tabularPrintingConstructing(s, field_names, ifNeedToAddTs1):

    x = PrettyTable()
    if ifNeedToAddTs1:
        x.field_names = ['ts1'] + field_names
    else:
        x.field_names = field_names
    for k in s.keys():
        if ifNeedToAddTs1:
            l = [k]
        else:
            l = []
        for kc in field_names:
            l.append(s[k][kc])
        x.add_row(l)
    return x


def tabularCvsDumping(fn, s, fieldnames, ifNeedToAddTs1):
    if ifNeedToAddTs1:
        augFieldnames = ['ts1'] + fieldnames
    else:
        augFieldnames = fieldnames
    with open(fn, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=augFieldnames)
        writer.writeheader()
        for k in s.keys():
            augSk = copy.deepcopy(s[k])
            augSk['ts1'] = k
            writer.writerow({key: augSk[key] for key in fieldnames})
