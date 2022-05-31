from ..py_ext.misc_v1 import mkdir_full
from shutil import rmtree
from collections import OrderedDict
import random
import os
import numpy as np
import torch
import pickle
from easydict import EasyDict


def compareEqualPair(A, B, **kwargs):  # A and B should be exactly the same, including the numpy vals
    if not (type(A) is type(B)):
        print('Type mismatch. Please investigate.')
        import ipdb
        ipdb.set_trace()
        raise ValueError('You cannot proceed.')
    if type(A) in [dict, EasyDict, OrderedDict]:
        if len(A.keys()) != len(B.keys()):
            print('dict key len does not match. Please investigate.')
            import ipdb
            ipdb.set_trace()
            raise ValueError('You cannot proceed.')
        for k in A.keys():
            if k not in B.keys():
                print('dict key does not match. Please investigate.')
                import ipdb
                ipdb.set_trace()
                raise ValueError('You cannot proceed.')
            assert compareEqualPair(A[k], B[k], **kwargs)
    elif type(A) in [list]:
        if len(A) != len(B):
            print('Len mismatch. Please investigate.')
            import ipdb
            ipdb.set_trace()
            raise ValueError('You cannot proceed.')
        for x, y in zip(A, B):
            assert compareEqualPair(x, y, **kwargs)
    elif type(A) in [np.ndarray, torch.Tensor]:
        if not (A.dtype == B.dtype):
            print('dtype mismatch between A and B.')
            import ipdb
            ipdb.set_trace()
            raise ValueError('You cannot proceed.')
        if 'float' in str(A.dtype):
            m = (A - B).abs().max()
            if m > kwargs['floatThre']:
                print('Float val of numpy array mismatch. Please investigate.')
                import ipdb
                ipdb.set_trace()
                raise ValueError('You cannot proceed.')
        elif ('int' in str(A.dtype)) or ('bool' in str(A.dtype)):
            if not (A == B).all():
                print('Int val of numpy array mismatch. Please investigate.')
                import ipdb
                ipdb.set_trace()
                raise ValueError('You cannot proceed.')
        else:
            print('Unknown np.adarray type: %s' % str(A.dtype))
            import ipdb
            ipdb.set_trace()
            raise ValueError('You cannot proceed.')
    else:
        if A != B:
            print('scalar value mismatch. Please investigate.')
            import ipdb
            ipdb.set_trace()
            raise ValueError('You cannot proceed.')
    # except:
    #     print('Exception')
    #     import ipdb
    #     ipdb.set_trace()
    #     print(1 + 1)
    return True


def compareConfigPair(loadedConfig, config):  # Note loadedConfig can be a subset of config
    assert type(loadedConfig) is type(config)
    if type(loadedConfig) in [dict, EasyDict, OrderedDict]:
        for k in loadedConfig.keys():
            if k not in config.keys():
                print('Missing key. Please investigate')
                import ipdb
                ipdb.set_trace()
                raise ValueError('You cannot proceed')
            assert compareConfigPair(loadedConfig[k], config[k])
    elif type(loadedConfig) in [list]:
        if len(loadedConfig) != len(config):
            print('Len mismatch. Please investigate.')
            import ipdb
            ipdb.set_trace()
            raise ValueError('You cannot proceed')
        for j in range(len(loadedConfig)):
            assert compareConfigPair(loadedConfig[j], config[j])
    else:
        if loadedConfig != config:
            print('Value mismatch. Please investigate.')
            import ipdb
            ipdb.set_trace()
            raise ValueError('You cannot proceed')
    return True


def checkPDSRLogDirNew(config, **kwargs):  # It is not functional! Might change the hard disk storage (mkdir stuffs)
    # return logDir

    # kwargs
    projRoot = kwargs['projRoot']  # Note projRoot should ends with '/'
    ifInitial = kwargs['ifInitial']  # required for any case

    P = config.P
    D = config.D
    S = config.S
    R = config.R

    # input checks
    assert P.startswith('P') and len(P) > 1 and '_' not in P
    assert D.startswith('D') and len(D) > 1 and '_' not in D
    assert S.startswith('S') and len(S) > 1 and '_' not in S
    assert R.startswith('R') and len(R) > 1 and '_' not in R
    assert projRoot.endswith('/')

    prefix = projRoot
    assert(os.path.isdir(prefix + 'v/P/%s/%s/' % (P, D)))  # the logDir must be manually created to the D level.
    logDir = prefix + 'v/P/%s/%s/%s/%s/' % (P, D, S, R)
    # Check logDir on the disk.
    if ifInitial:
        if R.startswith('Rtmp'):  # no matter whehter the logDir is existed or not, it will always be reset
            if os.path.isdir(logDir):
                rmtree(logDir)
        else:  # Will raise an error if the logDir is existed, and will make a new one if it is not existed.
            assert(not os.path.isdir(logDir))
        mkdir_full(logDir)
    else:  # Will always check if there is already an existed logDir. Will raise an error if there is none.
        assert(os.path.isdir(logDir))

    # Check config
    if os.path.isfile(logDir + 'config.pkl'):
        # assert os.path.isfile(logDir + 'config.txt')
        with open(logDir + 'config.pkl', 'rb') as f:
            loadedConfig = pickle.load(f)
        assert compareConfigPair(loadedConfig, config)
        '''
        # We do not need the following, because config might contain more due to new default value get in
        # for k in config.keys():
        #     assert config[k] == loadedConfig[k]
        for k in loadedConfig.keys():
            if k in ['R'] and loadedConfig['R'].startswith('Rtmp'):
                continue  # we waive this inconsistency
            if k in ['projDir']:
                continue  # temporary waive
            if type(config[k]) is np.ndarray:
                assert (config[k] == loadedConfig[k]).all()
            else:
                try:
                    assert (config[k] == loadedConfig[k]) or (np.isnan(config[k]) and np.isnan(loadedConfig[k]))
                except:
                    print('config inconsistent! Please check!')
                    import ipdb
                    ipdb.set_trace()
                    raise ValueError('You cannot proceed. It is dangerous to override.')
        '''

    else:
        assert not os.path.isfile(logDir + 'config.txt')
        # Do not override. Since you can always pass the check.
        # What we want to record in logger is to record the config that leads to the
        # model
        # If you need to full version - loaded from config.py!
        with open(logDir + 'config.pkl', 'wb') as f:
            pickle.dump(config, f)
        with open(logDir + 'config.txt', 'w') as f:
            for k in config.keys():
                f.write('%20s: %s\n' % (k, config[k]))

    mkdir_full(logDir + 'models/')
    mkdir_full(logDir + 'dump/')
    mkdir_full(logDir + 'cache/')
    mkdir_full(logDir + 'visTrain/')
    mkdir_full(logDir + 'visVal/')
    mkdir_full(logDir + 'numerical/')
    mkdir_full(logDir + 'monitorTrain/')
    mkdir_full(logDir + 'monitorVal/')
    return logDir


def save_network(log_dir, network, network_label, iter_label):
    assert os.path.isdir(log_dir)
    if not os.path.isdir(log_dir + 'models/'):
        mkdir_full(log_dir + 'models/')
    save_filename = 'models/{}_net_{}.pth'.format(network_label, iter_label)
    save_path = log_dir + save_filename
    if os.path.isfile(save_path) and ('latest' not in save_path):
        print('[save_network] Serious Warning: The snapshot dumping is already there. '
              'It seems that there might be a duplicate PDSR thread somewhere else. Please Check! ')
        import ipdb
        ipdb.set_trace()

    # torch.save(network.cpu().state_dict(), save_path)
    # network.cuda(device=cudaDevice)
    torch.save(network.state_dict(), save_path)  # just save the gpu version

    print('From save_network: Saving model to %s' % save_path)
    return

def probe_load_network(log_dir):
    iter_list = [int(x[x.rfind('_') + 1:x.find('.')])
                 for x in os.listdir(log_dir + 'models/')
                 if (x.endswith('.pth') or '.ckpt' in x) and ('latest' not in x)]
    if not iter_list:
        print(
            'From probe_load_network: did not find any existing network param snapshots, so makes no changes! Set resuming iter to be -1.')
        return -1
    max_iter = max(iter_list)
    resume_start_iter = max_iter
    print('From prob_load_network: Detected most recent model''s iter is %d' % resume_start_iter)
    return resume_start_iter

def load_network(log_dir, network, network_label, iter_label=None, map_location='cuda:0'):
    if iter_label is None:
        # iter_list = [int(x[x.rfind('_') + 1:-4]) for x in os.listdir(log_dir + 'models/') if x.endswith('.pth')]
        iter_list = [int(x[x.rfind('_') + 1:x.find('.')])
                     for x in os.listdir(log_dir + 'models/')
                     if x.endswith('.pth') or '.ckpt' in x]
        if not iter_list:
            print(
                'From load_network: did not find any existing network param snapshots, so makes no changes! Set resuming iter to be -1.')
            return -1, network
        max_iter = max(iter_list)
        iter_label = str(max_iter)
        resume_start_iter = max_iter
    else:
        resume_start_iter = int(iter_label)
    save_filename = 'models/{}_net_{}.pth'.format(network_label, iter_label)
    save_path = log_dir + save_filename
    network.load_state_dict(torch.load(save_path, map_location=map_location))
    print('From load_network: Loading model from %s' % save_path)
    return resume_start_iter, network

def castNP2THGPU(x_np):
    if type(x_np) is dict:
        x_thgpu = {}
        for k in x_np.keys():
            x_thgpu[k] = castNP2THGPU(x_np[k])
        return x_thgpu
    elif type(x_np) is list:
        return [castNP2THGPU(x) for x in x_np]
    elif type(x_np) is np.ndarray:
        return torch.from_numpy(x_np).cuda()


def castNP2THCPU(x_np):
    if type(x_np) is dict:
        x_thgpu = {}
        for k in x_np.keys():
            x_thgpu[k] = castNP2THCPU(x_np[k])
        return x_thgpu
    elif type(x_np) is list:
        return [castNP2THCPU(x) for x in x_np]
    elif type(x_np) is np.ndarray:
        return torch.from_numpy(x_np)


def castAnything(x, castCommand, **kwargs):
    device = kwargs.get('device', None)

    if type(x) is dict:
        return {k: castAnything(x[k], castCommand, **kwargs) for k in x.keys()}
    elif type(x) is list:
        return [castAnything(x0, castCommand, **kwargs) for x0 in x]
    elif type(x) is str or type(x) is int or type(x) is float:
        return x
    else:
        if castCommand == 'thgpu2np':
            assert type(x) is torch.Tensor and x.is_cuda
            return x.detach().cpu().numpy()
        elif castCommand == 'thgpu2thcpu':
            assert type(x) is torch.Tensor and x.is_cuda
            return x.detach().cpu()
        elif castCommand == 'thcpu2thgpu':
            assert type(x) is torch.Tensor and not x.is_cuda
            return x.cuda(device=device)
        elif castCommand == 'np2thgpu':
            assert type(x) is np.ndarray
            return torch.from_numpy(x).cuda(device=device)
        elif castCommand == 'np2thcpu':
            assert type(x) is np.ndarray
            return torch.from_numpy(x)
        elif castCommand == 'npOrThcpu2thcpu':
            assert (type(x) is np.ndarray) or (type(x) is torch.Tensor and not x.is_cuda)
            if type(x) is np.ndarray:
                return torch.from_numpy(x)
            elif type(x) is torch.Tensor and not x.is_cuda:
                return x
            else:
                raise ValueError('BUG!!!')
        elif castCommand == 'th2thcpu':
            assert type(x) is torch.Tensor
            return x.detach().cpu()
        elif castCommand == '2thcpu':
            assert (type(x) is np.ndarray) or (type(x) is torch.Tensor)
            if type(x) is np.ndarray:
                return torch.from_numpy(x)
            elif type(x) is torch.Tensor:
                return x.detach().cpu()
            else:
                raise ValueError('BUG!!!')
        elif castCommand == 'thcpu2np':
            assert type(x) is torch.Tensor and not x.is_cuda
            return x.numpy()
        else:
            raise NotImplementedError('Unknown combination of type(x): %s, and castCommand: %s' % (str(type(x)), castCommand))


def splitPDSRI(ts1):
    assert ts1.startswith('P')
    indP = 0
    indD = ts1.find('D')
    indS = ts1.find('S')
    indR = ts1.rfind('R')
    indI = ts1.rfind('I')
    assert indP < indD < indS < indR < indI
    P = ts1[indP:indD]
    D = ts1[indD:indS]
    S = ts1[indS:indR]
    R = ts1[indR:indI]
    I = ts1[indI:]
    return P, D, S, R, I


def splitPDSR(ts1):
    assert ts1.startswith('P')
    indP = 0
    indD = ts1.find('D')
    indS = ts1.find('S')
    indR = ts1.rfind('R')
    assert indP < indD < indS < indR
    P = ts1[indP:indD]
    D = ts1[indD:indS]
    S = ts1[indS:indR]
    R = ts1[indR:]
    return P, D, S, R


def splitPD(PD):
    assert PD.startswith('P')
    indP = 0
    indD = PD.find('D')
    assert indP < indD
    P = PD[indP:indD]
    D = PD[indD:]
    return P, D


def splitPDDrandom(PDDrandom):  # PDDrandom is typically a testSuiteName (PxDxDVisualizerName)
    assert PDDrandom.startswith('P')
    indP = 0
    indD = PDDrandom.find('D')
    indDrandom = PDDrandom[indD + 1:].find('D') + indD + 1
    assert indP < indD < indDrandom
    P = PDDrandom[indP:indD]
    D = PDDrandom[indD:indDrandom]
    Drandom = PDDrandom[indDrandom:]
    return P, D, Drandom


class DataLoaderKeeper(object):
    def __init__(self, dataLoaderList):
        self.dataLoaderList = dataLoaderList
        self.iterableDataLoderList = [iter(x) for x in dataLoaderList]
        self.len = len(dataLoaderList)

    def step(self):
        batch_thcpuList = []
        for j in range(self.len):
            try:
                tmp = next(self.iterableDataLoderList[j])
            except:
                self.iterableDataLoderList[j] = iter(self.dataLoaderList[j])
                tmp = next(self.iterableDataLoderList[j])
            batch_thcpuList.append(tmp)
        batch_thcpu = {
            k: torch.cat([x[k] for x in batch_thcpuList], 0) for k in batch_thcpuList[0].keys()
        }
        return batch_thcpu


class DatasetObjKeeper(object):
    def __init__(self, datasetObjList, ifTrain, batchSizeList):
        self.datasetObjList = datasetObjList
        self.len = len(datasetObjList)
        self.ifTrain = ifTrain
        self.batchSizeList = batchSizeList
        if not self.ifTrain:
            self.currentDatasetObjIndices = [0 for _ in range(self.len)]

    def step(self):
        batch_thcpuList = []
        for j in range(self.len):
            for _ in range(self.batchSizeList[j]):
                if self.ifTrain:
                    index = random.randint(0, len(self.datasetObjList[j]) - 1)
                else:
                    index = self.currentDatasetObjIndices[j]
                    self.currentDatasetObjIndices[j] += 1
                batch_thcpuList.append(self.datasetObjList[j][index])
        batch_thcpu = {
            k: torch.stack([x[k] for x in batch_thcpuList], 0) for k in batch_thcpuList[0].keys()
        }
        return batch_thcpu


def constructInitialBatchStepVis0(batch_vis, **kwargs):
    iterCount = kwargs['iterCount']
    visIndex = kwargs['visIndex']
    dataset = kwargs['dataset']
    P = kwargs['P']
    D = kwargs['D']
    S = kwargs['S']
    R = kwargs['R']
    methodologyName = '%s%s%s%sI%d' % (P, D, S, R, iterCount)
    if kwargs['verboseGeneral'] > 0:
        print('[Visualizer] constructInitialBatchStepVis0 (iterCount=%s, P=%s, D=%s, S=%s, R=%s)' %
              (iterCount, P, D, S, R))
    from datasets_registration import datasetRetrieveList
    if dataset is None:
        _dataset = ''
    else:
        _dataset = '_' + dataset
    bsv0 = {
        'iterCount': iterCount,
        'visIndex': visIndex,  # must contain (always 0)
        'P': P, 'D': D, 'S': S, 'R': R,
        'methodologyName': methodologyName,  # must contain
        'index': int(batch_vis['index%s' % _dataset][visIndex]),  # must contain
        'did': int(batch_vis['did%s' % _dataset][visIndex]),  # must contain (always 0)
        'datasetID': int(batch_vis['datasetID%s' % _dataset][visIndex]),  # must contain
        'dataset': datasetRetrieveList[int(batch_vis['datasetID%s' % _dataset][visIndex])],  # must contain
        'flagSplit': int(batch_vis['flagSplit%s' % _dataset][visIndex]),  # must contain
    }
    return bsv0


def mergeFromBatchVis(bsv0, batch_vis, **kwargs):
    dataset = kwargs['dataset']
    visIndex = kwargs['visIndex']

    existingKeys = ['iterCount', 'visIndex', 'P', 'D', 'S', 'R', 'methodologyName',
                    'index', 'did', 'datasetID', 'dataset', 'flagSplit']
    for k_dataset in batch_vis.keys():
        ss = k_dataset.split('_')
        assert len(ss) <= 2
        if dataset is None:
            if len(ss) == 1:
                k = k_dataset
                if k not in existingKeys:
                    assert k not in bsv0.keys()
                    if batch_vis[k].ndim > 0:
                        bsv0[k] = batch_vis[k][visIndex]
        else:
            if len(ss) == 2 and ss[1] == dataset:
                k = ss[0]
                if k not in existingKeys:
                    assert k not in bsv0.keys()
                    if batch_vis[k_dataset].ndim > 0:
                        bsv0[k] = batch_vis[k_dataset][visIndex]
    return bsv0


def mergeFromAnotherBsv0(bsv0, anotherBsv0, **kwargs):
    # visIndex = bsv0['visIndex']
    existingKeys = ['iterCount', 'visIndex', 'P', 'D', 'S', 'R', 'methodologyName',
                    'index', 'did', 'datasetID', 'dataset', 'flagSplit']
    copiedKeys = kwargs['copiedKeys']
    assert type(copiedKeys) is list
    for k in copiedKeys:  # anotherBsv0.keys():
        if k not in existingKeys:
            assert k not in bsv0.keys()
            if anotherBsv0[k].ndim > 0:
                bsv0[k] = anotherBsv0[k]
    return bsv0
