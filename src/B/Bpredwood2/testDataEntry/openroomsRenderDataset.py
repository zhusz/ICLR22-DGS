# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


class OpenroomsVoxCache(object):
    def __init__(self, **kwargs):
        self.openroomsVoxCache = {}
        projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../../'
        self.loading_root = projRoot + 'v/R/openrooms/R11b/'
        assert os.path.isdir(self.loading_root), self.loading_root

    def call_cache_openrooms_vox_0(self, **kwargs):
        houseID0 = int(kwargs['houseID0'])
        verbose = kwargs['verbose']

        if houseID0 not in self.openroomsVoxCache.keys():
            t = time.time()
            raise NotImplementedError("TODO when you are getting them into the training part.")
            # with open()
