# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# see how the depth_zbuffer scaled from uint16 to float using the standard data loader
import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
sys.path.append(projRoot + 'src/B/')
from BpredwoodA.testDataEntry.omnidataBerkeley.omnidataBerkeley import OmnidataBerkeleyDataset


def main():
    datasetConf = {
        'dataset': 'omnidataBerkeley', 'trainingSplit': 'train', 'batchSize': 8,
        'class': 'OmnidataBerkeleyDataset',

        'componentFrequency': {
            'taskonomy': 0.5, 'replica': 0.1, 'gso': 0.05,
            'hypersim': 0.3, 'blendedMVS': 0.05,
        },

        'winWidth': 384, 'winHeight': 384,
    }

    datasetObj = OmnidataBerkeleyDataset(
        datasetConf, projRoot=projRoot, componentName='taskonomy',
        datasetSplit='train', ifConductDataAugmentation=False,
    )
    b0np = datasetObj.getOneNP(0)

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()
