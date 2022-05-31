# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from UDLv3 import udl
import numpy as np
from datasets_registration import datasetDict
import random
from .transforms import default_loader, get_transform
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


bt = lambda s: s[0].upper() + s[1:]


class OmnidataBerkeleyDataset(object):
    def __init__(self, datasetConf, **kwargs):
        # basics
        self.meta = {}

        # kwargs
        self.projRoot = kwargs['projRoot']
        self.datasetSplit = kwargs['datasetSplit']
        self.componentName = kwargs['componentName']
        self.ifConductDataAugmentation = kwargs['ifConductDataAugmentation']

        # datasetConf
        self.datasetConf = datasetConf

        # layout variables
        dataset = datasetConf['dataset']
        self.dataset = dataset
        assert dataset == 'omnidataBerkeley'
        assert self.componentName in ['taskonomy', 'replica', 'gso', 'hypersim', 'blendedMVS']

        # preset
        self.raw_root = self.projRoot + 'remote_fastdata/omnidata/%s/' % \
                        ('ta' if self.componentName == 'taskonomy' else 'om')

        # udl
        A1 = udl('pkl_A1_', dataset)
        flagSplit = A1['flagSplit'].copy()
        componentIDList = A1['componentIDList']
        componentNameIDMapDict = A1['componentNameIDMapDict']
        flagSplit[componentIDList != componentNameIDMapDict[self.componentName]] = 0
        self.flagSplit = flagSplit
        self.indTrain = np.where(self.flagSplit == 1)[0]
        self.indVal = np.where(self.flagSplit == 2)[0]
        self.indTest = np.where(self.flagSplit == 3)[0]
        self.mTrain = int(len(self.indTrain))
        self.mVal = int(len(self.indVal))
        self.mTest = int(len(self.indTest))
        self.buildingNameList = A1['buildingNameList']
        self.pointIDList = A1['pointIDList']
        self.viewIDList = A1['viewIDList']

        # datasetCaches
        self.datasetCaches = {}

        # copied from omnidata_tools
        self.tasks = ['rgb', 'depth_zbuffer', 'mask_valid']
        self.transform = {task: get_transform(task, image_size=None) for task in self.tasks}
        # We do not do rgb normalization in dataLoaders (we do in image encoders instead)
        self.componentNameToPathComponentName = {
            'taskonomy': 'taskonomy',
            'replica': 'replica',
            'gso': 'replica_gso',
            'hypersim': 'hypersim',
            'blendedMVS': 'blended_mvg',
        }

    def __len__(self):
        if self.datasetSplit == 'train':
            return self.mTrain
        elif self.datasetSplit == 'val':
            return self.mVal
        elif self.datasetSplit == 'test':
            return self.mTest
        else:
            raise NotImplementedError('Unknown self.datasetSplit: %s' % self.datasetSplit)

    def getOneNP(self, index):
        # layout variables
        datasetConf = self.datasetConf
        dataset = datasetConf['dataset']

        componentNameToPathComponentName = self.componentNameToPathComponentName
        ifConductDataAugmentation = self.ifConductDataAugmentation
        componentName = self.componentName
        assert datasetConf['winWidth'] == datasetConf['winHeight']
        imageSize = datasetConf['winHeight']

        buildingName = self.buildingNameList[index]
        pointID = self.pointIDList[index]
        viewID = self.viewIDList[index]

        b0np = dict(
            index=np.array(index, dtype=np.int32),
            dataset=dataset,
            flagSplit=np.array(self.flagSplit[index], dtype=np.float32),
        )

        b0np['componentName'] = componentName
        b0np['buildingName'] = buildingName
        b0np['pointID'] = np.array(pointID, dtype=np.int32)
        b0np['viewID'] = np.array(viewID, dtype=np.int32)

        if datasetConf['ifDummy']:
            winWidth, winHeight = datasetConf['winWidth'], datasetConf['winHeight']
            b0np['depthZBuffer'] = np.random.rand(1, winHeight, winWidth).astype(np.float32)
            b0np['maskValid'] = np.ones((1, winHeight, winWidth)).astype(np.float32)
            b0np['rgb'] = np.random.rand(3, winHeight, winWidth).astype(np.float32)
            return b0np

        # copied from omnidata_tools

        if ifConductDataAugmentation:
            flip = random.random() > 0.5
        else:
            flip = False
        for task_num, task in enumerate(self.tasks):
            # path = self.url_dict[(task, building, point, v)]
            path = self.raw_root + '%s/%s/%s/point_%d_view_%d_domain_%s.png' \
                   % (task, componentNameToPathComponentName[componentName],
                      buildingName, pointID, viewID, task)
            if (task == 'mask_valid') and (componentName == 'taskonomy'):
                res = np.array(
                    (b0np['depthZBuffer'] < ((2 ** 16 - 1.) / 8000. * 0.98)) &
                    (b0np['depthZBuffer'] > 0.01)
                , dtype=np.float32)
            else:
                resize_method = Image.BILINEAR if task in ['rgb'] else Image.NEAREST
                res = default_loader(path)

                if ifConductDataAugmentation and (
                        path.__contains__('hypersim') or path.__contains__('BlendedMVS')):
                    resize_transform = transforms.Resize(imageSize, resize_method)
                    res = resize_transform(res)
                    if task_num == 0:
                        i, j, h, w = transforms.RandomCrop.get_params(
                            res, output_size=(imageSize, imageSize))
                    res = TF.crop(res, i, j, h, w)
                    res = self.transform[task](res)
                else:
                    transform = transforms.Compose([
                        transforms.Resize(imageSize, resize_method),
                        transforms.CenterCrop(imageSize),
                        self.transform[task]])
                    res = transform(res)
                if flip:
                    res = torch.flip(res, [2])
                    if task == 'normal': res[0,:,:] = 1 - res[0,:,:]
                res = res.numpy()
            if task == 'depth_zbuffer':
                b0np['depthZBuffer'] = res
            elif task == 'mask_valid':
                b0np['maskValid'] = res
            else:
                b0np[task] = res

        return b0np

    def __getitem__(self, indexTrainValTest):
        overfitIndexTvt = self.datasetConf.get('singleSampleOverfittingIndexTvt', 0)

        if self.datasetSplit == 'train':
            index = self.indTrain[indexTrainValTest]
            overfitIndex = self.indTrain[overfitIndexTvt]
        elif self.datasetSplit == 'val':
            index = self.indVal[indexTrainValTest]
            overfitIndex = self.indVal[overfitIndexTvt]
        elif self.datasetSplit == 'test':
            index = self.indTest[indexTrainValTest]
            overfitIndex = self.indTest[overfitIndexTvt]
        else:
            print('Warning: Since your datasetSplit is %s, you cannot call this function!' %
                  self.datasetSplit)
            raise ValueError

        if self.datasetConf.get('singleSampleOverfittingMode', False):
            index = int(overfitIndex)
        else:
            index = int(index)

        b0np = self.getOneNP(index)
        return b0np
