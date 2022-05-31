# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import time
import socket
import GPUtil


if socket.gethostname() in ['x1', 'x2']:
    torchGpuId2NvidiaSmiGpuId = [0, 1, 4, 7, 8, 2, 3, 5, 6, 9]
elif socket.gethostname() in ['y1', 'y2', 'y3', 'y4'] or socket.gethostname().startswith('y'):
    torchGpuId2NvidiaSmiGpuId = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
else:
    raise NotImplementedError('Unknown socket.gethousename(): %s' % (socket.gethostname()))
device = 'cuda:0'
a = []
torchGpuId = int(os.environ['CUDA_VISIBLE_DEVICES'])
print('Memory Forwarder on GPU %d' % torchGpuId)
while True:
    time.sleep(1)
    if GPUtil.getGPUs()[torchGpuId2NvidiaSmiGpuId[torchGpuId]].memoryFree >= 500:  # (~0.5G)
        try:
            b = torch.zeros(10000, 10000).float().to(device)
            a.append(b)
            print('Successfully forward the GPU %d memory one time (Remaining Memory is %.1f). len(a) now is %d. Current time is %s.' %
                  (torchGpuId, GPUtil.getGPUs()[torchGpuId2NvidiaSmiGpuId[torchGpuId]].memoryFree, len(a), time.localtime()))
        except:
            print('GPU %d has free memory of %.1f but cannot allocate. Please check! Current time is %s.' %
                  (torchGpuId, GPUtil.getGPUs()[torchGpuId2NvidiaSmiGpuId[torchGpuId]].memoryFree, time.localtime()))
