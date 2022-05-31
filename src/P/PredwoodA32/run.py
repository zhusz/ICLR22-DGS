# (tfconda)

# Single thread training:
#   D=DX S=SX R=RX CUDA_VISIBLE_DEVICES=0 op -m PX.run

# Multi-thread training:
#   D=DX S=SX R=RX CUDA_VISIBLE_DEVICES=0,1,2,3 op -m torch.distributed.launch \
#       --master_port=X --nproc_per_node=4 --use_env -m PX.run

import os
import sys
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'
sys.path.append(projRoot + 'src/versions/')
sys.path.append(projRoot + 'src/B/')
from configs_registration import getConfigGlobal
from codes_py.toolbox_3D.pyrender_wrapper_v1 import PyrenderManager


def main():
    fullPathSplit = os.path.dirname(os.path.realpath(__file__)).split('/')
    P = fullPathSplit[-1]
    D = os.environ['D']
    S = os.environ['S']
    R = os.environ['R']
    getConfigFunc = getConfigGlobal(P, D, S, R)['getConfigFunc']
    config = getConfigFunc(P, D, S, R)

    DTrainer = getConfigGlobal(P, D, S, R)['exportedClasses']['DTrainer']

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        numMpProcess = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        numMpProcess = 0

    trainer = DTrainer(config, rank=rank, numMpProcess=numMpProcess, ifDumpLogging=True)
    print(trainer.config)

    if 'I' in list(os.environ.keys()):
        iter_label = int(os.environ['I'][1:])
    else:
        iter_label = None

    trainer.initializeAll(iter_label=iter_label, hook_type=None, ifLoadToCuda=True)
    trainer.train(pyrenderManager=PyrenderManager(256, 256))


if __name__ == '__main__':
    main()


# Additional installations
# ninja
# moderngl (corenet)
