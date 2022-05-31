# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (tfconda)
# run under src/A, run python -m omnidataBerkeley.A1_m
import os
from collections import OrderedDict
import numpy as np
import sys
import pickle
from .data_omnidata_tools.omnidata_dataset import OmnidataDataset


def main():
    projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
    dataset = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    omnidata_tools_raw_root = projRoot + 'remote_fastdata/omnidata_tools/'
    valsets = OrderedDict(
        (omnidata_tool_dataset, OmnidataDataset(
            options=(
                OmnidataDataset.Options(
                    tasks=['rgb', 'depth_zbuffer', 'mask_valid'],
                    datasets=[omnidata_tool_dataset],
                    split='val',
                    taskonomy_variant='fullplus',
                    transform='DEFAULT',
                    image_size=384,
                    normalize_rgb=True,
                    normalization_mean=[0., 0., 0.],
                    normalization_std=[1., 1., 1.],
                    taskonomy_data_path=omnidata_tools_raw_root + 'taskonomy',  # no /
                    replica_data_path=omnidata_tools_raw_root + 'replica',  # no /
                    gso_data_path=omnidata_tools_raw_root + 'replica_gso',  # no /
                    hypersim_data_path=omnidata_tools_raw_root + 'hypersim',  # no /
                    blendedMVS_data_path=omnidata_tools_raw_root + 'blended_mvg',  # no /
                )
            )
        ))
        for omnidata_tool_dataset in [
            'taskonomy', 'replica', 'gso', 'hypersim', 'blendedMVS']  # 0, 1, 2, 3, 4
    )
    trainsets = OrderedDict(
        (omnidata_tool_dataset, OmnidataDataset(
            options=(
                OmnidataDataset.Options(
                    tasks=['rgb', 'depth_zbuffer', 'mask_valid'],
                    datasets=[omnidata_tool_dataset],
                    split='train',
                    taskonomy_variant='fullplus',
                    transform='DEFAULT',
                    image_size=384,
                    normalize_rgb=True,
                    normalization_mean=[0., 0., 0.],
                    normalization_std=[1., 1., 1.],
                    taskonomy_data_path=omnidata_tools_raw_root + 'taskonomy',  # no /
                    replica_data_path=omnidata_tools_raw_root + 'replica',  # no /
                    gso_data_path=omnidata_tools_raw_root + 'replica_gso',  # no /
                    hypersim_data_path=omnidata_tools_raw_root + 'hypersim',  # no /
                    blendedMVS_data_path=omnidata_tools_raw_root + 'blended_mvg',  # no /
                )
            )
        ))
        for omnidata_tool_dataset in [
            'taskonomy', 'replica', 'gso', 'hypersim', 'blendedMVS']  # 0, 1, 2, 3, 4
    )
    componentNameList = []  # e.g. 'taskonomy'
    componentIDList = []  # 0, 1, 2, 3, 4
    componentNameIDMapDict = {}
    componentIDNameMapRetrieveList = {}
    buildingNameList = []  # e.g. 'almota'
    pointIDList = []
    viewIDList = []
    flagSplit = []
    for i, s in enumerate([trainsets, valsets]):
        for o, omnidata_tool_dataset in enumerate([
                'taskonomy', 'replica', 'gso', 'hypersim', 'blendedMVS']):
            if i == 0:
                componentNameIDMapDict[omnidata_tool_dataset] = o
                componentIDNameMapRetrieveList[o] = omnidata_tool_dataset
            componentNameList += ([omnidata_tool_dataset] * len(s[omnidata_tool_dataset]))
            componentIDList += ([o] * len(s[omnidata_tool_dataset]))
            buildingNameList += [x[0] for x in s[omnidata_tool_dataset].bpv_list]
            pointIDList += [int(x[1]) for x in s[omnidata_tool_dataset].bpv_list]
            viewIDList += [int(x[2]) for x in s[omnidata_tool_dataset].bpv_list]
            flagSplit += ([i + 1]) * len(s[omnidata_tool_dataset])
    componentIDList = np.array(componentIDList, dtype=np.int32)
    pointIDList = np.array(pointIDList, dtype=np.int32)
    viewIDList = np.array(viewIDList, dtype=np.int32)
    flagSplit = np.array(flagSplit, dtype=np.int32)
    m = len(componentNameList)

    with open(projRoot + 'v/A/%s/A1_m.pkl' % dataset, 'wb') as f:
        pickle.dump({
            'm': m,
            'componentNameList': componentNameList,
            'componentIDList': componentIDList,
            'componentNameIDMapDict': componentNameIDMapDict,
            'componentIDNameMapRetrieveList': componentIDNameMapRetrieveList,
            'buildingNameList': buildingNameList,
            'pointIDList': pointIDList,
            'viewIDList': viewIDList,
            'flagSplit': flagSplit,
        }, f)

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()
