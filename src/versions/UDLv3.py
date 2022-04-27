# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from codes_py.np_ext.mat_io_v1 import pLoadMat, pProcessMat
from codes_py.np_ext.np_image_io_v1 import imread as my_imread
from codes_py.py_ext.misc_v1 import cellstr2liststr
from scipy.io import loadmat
from skimage.io import imread as sk_imread
from easydict import EasyDict
import pickle
import json
import numpy as np
import os


v_root = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../v/'


def udl(tag, dataset=None, index=None):
    # mat(default) / rgb / segUniProb / mats (each sample has a mat, in python it is a dict of np.ndarray)
    # quiry format: _F1b_openpose18 or _F1b_ or rgb_A1_img or _Z_trisSmpl0

    # provided annotations or synthesized labels or sample-specific meta (e.g. idxA and idxB for the pairwise dataset)
    # A registration (unregistered are all regarded as mat)
    # all intermediate results (including PDSR results)
    # B-Y registration (unregistred are all regarded as mat)
    # meta (not belonging to any dataset, and not correspond to indexing)
    # Z registration. example: (tag='_Z_smplTris', dataset=None, index=None)

    ts = tag.split('_')
    assert len(ts) == 3
    if len(ts[0]) == 0:
        ts[0] = 'mat'
    if ts[1].startswith('Z') or ts[1].startswith('P'):
        assert dataset is None
        assert index is None
    else:
        assert dataset is not None

    if ts[0] not in ['mat', 'pkl', 'mats', 'pkls', 'jsons', 'jpg', 'png']:
        assert len(ts[2]) > 0

    if ts[1].startswith('Z'):  # ts[1] == 'Z' indicates that ts[0] can be arbitrary stuff (typically void)
        assert ts[1] == 'Z'
        if ts[2] == 'smplTorchTxt':
            with open(v_root + 'Z/smplTorch/neutral_smpl_with_cocoplus_reg.txt', 'r') as f:
                model = json.load(f)
            return model
        if ts[2] == 'smplTris':
            smplTris = np.load(v_root + 'Z/smplTris.npy').astype(np.int32)
            return smplTris
        if ts[2] in ['unionCategory', 'repCategory', 'mpCategory']:
            with open(v_root + 'Z/semanticRepMp/%s.pkl' % ts[2], 'rb') as f:
                pkl = pickle.load(f)
            return pkl
        if ts[2] in ['semanticNYU']:
            mat = pLoadMat(v_root + 'Z/semanticNYU/semanticNYU.mat')
            with open(v_root + 'Z/semanticNYU/semanticNYU.pkl', 'rb') as f:
                pkl = pickle.load(f)
            for k in pkl.keys():
                mat[k] = pkl[k]
            return mat
        if ts[2] in ['semanticREP']:
            mat = pLoadMat(v_root + 'Z/semanticNYU/semanticREP.mat')
            with open(v_root + 'Z/semanticNYU/semanticREP.pkl', 'rb') as f:
                pkl = pickle.load(f)
            for k in pkl.keys():
                mat[k] = pkl[k]
            return mat
        if ts[2] in ['semanticMP']:
            mat = pLoadMat(v_root + 'Z/semanticNYU/semanticMP.mat')
            with open(v_root + 'Z/semanticNYU/semanticMP.pkl', 'rb') as f:
                pkl = pickle.load(f)
            for k in pkl.keys():
                mat[k] = pkl[k]
            return mat
        if ts[2] in ['semanticPIX3D']:
            mat = pLoadMat(v_root + 'Z/semanticNYU/semanticPIX3D.mat')
            with open(v_root + 'Z/semanticNYU/semanticPIX3D.pkl', 'rb') as f:
                pkl = pickle.load(f)
            for k in pkl.keys():
                mat[k] = pkl[k]
            return mat
        if ts[2] in ['semanticFront']:
            with open(v_root + 'Z/semanticFront/semanticFront.pkl', 'rb') as f:
                pkl = pickle.load(f)
            return pkl
        if ts[2] == 'cate':
            with open(v_root + 'Z/cate/cate.pkl', 'rb') as f:
                pkl = pickle.load(f)
            return pkl
        else:
            raise NotImplementedError('Z: Unrecognized Z fetching %s' % ts[2])

    if ts[0] in ['jpg', 'png']:  # with soft link  (the saving coutnerpart can be arbitrary)
        # Do not have any assumption on the images to read, so to load from np_ext.np_image_io.imread
        # For the images to read, completely no assumptions on size / range / dtype / file_ext
        # Fot the return images, always promise to be the same as 'rgb'
        # slow but can directly derived from the original dataset (e.g. jpg files)
        assert type(index) is int
        return my_imread(v_root + '%s/' % ts[1][0] + dataset + '/' + ts[1] + '/' + '%08d.%s' % (index, ts[0]))
    elif ts[0] == 'seg':  # saving counterpart: imsave(.png, 1-channel, np.uint8)  -> so stored as np.uint8
        # no check as above
        # the obtained seg is always assumed to be H * W, uint8 0-255, with PNG encoding
        raise NotImplementedError
    elif ts[0] == 'jsons':
        assert type(index) is int
        with open(v_root + 'A/' + dataset + '/' + ts[1] + '/' + '%08d.json' % index, 'r') as f:
            j = json.load(f)
        if len(ts[2]) == 0:
            return j
        else:
            return j[ts[2]]
    elif ts[0] == 'mats':
        assert type(index) is int
        d = loadmat(v_root + '%s/' % ts[1][0] + dataset + '/' + ts[1] + '/' + '%08d.mat' % index)
        pProcessMat(d)
        if len(ts[2]) == 0:
            return d  # should be a dict
        else:
            return d[ts[2]]
    elif ts[0] == 'pkls':
        assert type(index) is int
        with open(v_root + '%s/' % ts[1][0] + dataset + '/' + ts[1] + '/' + '%08d.pkl' % index, 'rb') as f:
            p = pickle.load(f)
        if len(ts[2]) == 0:
            return p  # should be a dict
        else:
            return p[ts[2]]
    elif ts[0] == 'pkl':
        def pkl_fetch(pkl_root):

            pkl_fn_candidates = os.listdir(pkl_root)
            pkl_fn_right_candidates = [x for x in pkl_fn_candidates
                                       if x.startswith(ts[1] + '_') and x.endswith('.pkl')]
            if len(pkl_fn_right_candidates) != 1:
                print('# of right candidate wrong! %d instead of 1 for ts1 %s!' % (len(pkl_fn_right_candidates), ts[1]))
                import ipdb
                ipdb.set_trace()
                raise ValueError('# of right candidate wrong! %d instead of 1 for ts1 %s!' % (len(pkl_fn_right_candidates), ts[1]))
            # assert(len(pkl_fn_right_candidates) == 1)
            pkl_fn = pkl_fn_right_candidates[0]
            with open(pkl_root + pkl_fn, 'rb') as f:
                pkl = pickle.load(f)
            # No need for special process as all are in python

            assert type(pkl) is dict
            if len(ts[2]) == 0:
                assert index is None
                return pkl
            elif index is None:
                return pkl[ts[2]]
            else:  # index can be int / list of ints / np.ndarray (np.int32)
                return pkl[ts[2]][index]
        return pkl_fetch(v_root + ts[1][0] + '/' + dataset + '/')
    elif ts[0] == 'mat':
        def mat_fetch(mat_root):

            mat_fn_candidates = os.listdir(mat_root)
            mat_fn_right_candidates = [x for x in mat_fn_candidates
                                       if x.startswith(ts[1] + '_') and x.endswith('.mat')]
            if len(mat_fn_right_candidates) != 1:
                print('# of right candidate wrong! %d instead of 1 for ts1 %s! (dataset: %s)' % (len(mat_fn_right_candidates), ts[1], dataset))
                raise ValueError('# of right candidate wrong! %d instead of 1 for ts1 %s! (dataset: %s)' % (len(mat_fn_right_candidates), ts[1], dataset))
            # assert(len(mat_fn_right_candidates) == 1)
            mat_fn = mat_fn_right_candidates[0]
            mat = loadmat(mat_root + mat_fn)

            pProcessMat(mat)
            if len(ts[2]) == 0:
                assert index is None
                return mat
            elif index is None:
                return mat[ts[2]]
            else:
                # index can be int / list of ints / np.ndarray (np.int32)
                return mat[ts[2]][index]
        return mat_fetch(v_root + '%s/' % ts[1][0] + dataset + '/')
    else:
        raise NotImplementedError('tag: %s, dataset: %s, index: %s' % (tag, dataset, index))
