# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from scipy.io import savemat
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.nn import init
import functools


def _get_numpy_size(t):  # t is a pytorch tensor
    tt = np.zeros((len(t.shape)))
    for i in range(len(t.shape)):
        tt[i] = t.shape[i]
    return tt


# pth model to mat model transformer
def pthModel2MatModel(pthFile, matFile, modelName, ifDetail=False):
    model = torch.load(pthFile)
    if ifDetail:
        a = np.zeros((len(model) + 1, 8), dtype=object)
        a[0, :] = [modelName, 'shape', 'mean', 'meanabs', 'min', 'max', 'std', 'detail']
    else:
        a = np.zeros((len(model) + 1, 7), dtype=object)
        a[0, :] = [modelName, 'shape', 'mean', 'meanabs', 'min', 'max', 'std']
    for j in range(len(model)):
        a[j + 1, 0] = model.keys()[j]
        c = model[a[j + 1, 0]]
        a[j + 1, 1] = _get_numpy_size(c)  # str(c.shape)
        if 'float' in str(c.dtype):
            a[j + 1, 2] = float(c.mean())
            a[j + 1, 3] = float(np.abs(c).mean())
            a[j + 1, 4] = float(c.min())
            a[j + 1, 5] = float(c.max())
            a[j + 1, 6] = float(c.std())
        else:
            a[j + 1, 2] = c.numpy()
        if ifDetail:
            a[j + 1, 7] = c.numpy()
    savemat(matFile, {'model': a})


# Define hookers
# hooker_output_list = []
# if you want to reuse this hooker, make sure to do the following
# 1. delete the previous hooker by calling remove_all_hooks
# 2. reset hooker_output_list to be []

class PyTorchForwardHook(object):
    def __init__(self, content_to_save, module_top, module_name):
        self.content_to_save = content_to_save
        self.module_top = module_top
        self.module_name = module_name

        self.h_list = []

        if self.content_to_save == 'full':
            header = np.zeros(9, dtype=object)
            header[:] = [module_name, 'name', 'size', 'mean', 'meanabs', 'min', 'max', 'std', 'output']
        elif self.content_to_save == 'stat':
            header = np.zeros(8, dtype=object)
            header[:] = [module_name, 'name', 'size', 'mean', 'meanabs', 'min', 'max', 'std']
        self.hooker_output_list = [header]

        header = np.zeros(29, dtype=object)
        header[:] = [module_name, 'weight_size', 'mean', 'meanabs', 'min', 'max', 'std', 'weight', 'bias_size', 'mean', 'meanabs', 'min', 'max', 'std', 'bias', 'runningMean_size', 'mean', 'meanabs', 'min', 'max', 'std', 'runningMean', 'runningVar_size', 'mean', 'meanabs', 'min', 'max', 'std', 'runningVar']  # It is not big so we always stick to 'full'
        self.hooker_params_list = [header]

        self._construct_forward_hook(module_top, [], self.h_list)

    def __forward_full_hook_fun(self, m, i, o):  # hook func cannot export =, because it just change the temparoray reference! You should first take 61A.
        t = torch.FloatTensor(o.data.shape)
        t.copy_(o.data)
        t1 = np.zeros((5))
        for i in range(len(t.shape)):
            t1[i] = t.shape[i]
        self.hooker_output_list.append([str(m), m.hookParentName, t1, float(t.mean()), float(t.abs().mean()), float(t.min()), float(t.max()), float(t.std()), t.detach().cpu().numpy()])

    def __forward_stat_hook_fun(self, m, i, o):
        t1 = np.zeros((5))  # size0-3
        t_light = o.data
        for i in range(len(t_light.shape)):
            t1[i] = t_light.shape[i]
        self.hooker_output_list.append([str(m), m.hookParentName, t1, float(t_light.mean()), float(t_light.abs().mean()), float(t_light.min()), float(t_light.max()), float(t_light.std())])

    def _construct_forward_hook(self, module_now, stack_str_list, h_list):
        if len(module_now._modules.keys()) > 0:
            # Process composite modules, no actually storage from here
            count = 0
            for sub_module_tag in module_now._modules.keys():
                sub_module = module_now._modules.get(sub_module_tag)
                if hasattr(module_now, 'names'):  # mainly for ModuleList
                    stack_str_list.append(module_now.names[count])
                else:
                    stack_str_list.append(sub_module_tag)
                self._construct_forward_hook(sub_module, stack_str_list, h_list)
                stack_str_list.pop()
                count += 1
        else:
            # Process bottom modules, do all the storage registering
            s = str(module_now)
            t1 = torch.FloatTensor()
            t2 = torch.FloatTensor()
            t3 = torch.FloatTensor()
            t4 = torch.FloatTensor()
            # param storing
            if s.find('Conv') >= 0 or s.find('Norm') >= 0 or s.find('Linear') >= 0:
                if hasattr(module_now, 'bias') and (module_now.bias is not None):
                    t1.resize_(module_now.weight.data.shape)
                    t1.copy_(module_now.weight.data)
                    t2.resize_(module_now.bias.data.shape)
                    t2.copy_(module_now.bias.data)
                    if s.find('BatchNorm') >= 0:
                        t3.resize_(module_now.running_mean.data.shape)
                        t3.copy_(module_now.running_mean.data)
                        t4.resize_(module_now.running_var.data.shape)
                        t4.copy_(module_now.running_var.data)
                        p = [_get_numpy_size(t1), float(t1.mean()), float(t1.abs().mean()), float(t1.min()), float(t1.max()), float(t1.std()), t1.detach().cpu().numpy(), _get_numpy_size(t2), float(t2.mean()), float(t2.abs().mean()), float(t2.min()), float(t2.max()), float(t2.std()), t2.detach().cpu().numpy(), _get_numpy_size(t3), float(t3.mean()), float(t3.abs().mean()), float(t3.min()), float(t3.max()), float(t3.std()), t3.detach().cpu().numpy(), _get_numpy_size(t4), float(t4.mean()), float(t4.abs().mean()), float(t4.min()), float(t4.max()), float(t4.std()), t4.detach().cpu().numpy()]
                    else:
                        p = [_get_numpy_size(t1), float(t1.mean()), float(t1.abs().mean()), float(t1.min()), float(t1.max()), float(t1.std()), t1.detach().cpu().numpy(), _get_numpy_size(t2), float(t2.mean()), float(t2.abs().mean()), float(t2.min()), float(t2.max()), float(t2.std()), t2.detach().cpu().numpy()]
                        for i in range(14):
                            p.append([])
                else:
                    t1.resize_(module_now.weight.data.shape)
                    t1.copy_(module_now.weight.data)
                    p = [_get_numpy_size(t1), float(t1.mean()), float(t1.abs().mean()), float(t1.min()), float(t1.max()), float(t1.std()), t1.detach().cpu().numpy()]
                    for i in range(21):
                        p.append([])
            else:
                p = []
                for i in range(28):
                    p.append([])
            p = ['.'.join(stack_str_list)] + p
            self.hooker_params_list.append(p)
            # register outputs. The first line writes the headings
            module_now.hookParentName = '.'.join(stack_str_list)
            if self.content_to_save == 'full':
                h_list.append(module_now.register_forward_hook(self.__forward_full_hook_fun))
            elif self.content_to_save == 'stat':
                h_list.append(module_now.register_forward_hook(self.__forward_stat_hook_fun))
            else:
                raise NotImplementedError('only full and stat avialble for content_to_save.')

    def remove_all_hooks(self):  # We would recommend not to reuse the same object of this class on two different top modules
        for h in self.h_list:
            h.remove()
