# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (tfconda)
import os
from codes_py.py_ext.misc_v1 import mkdir_full
from subprocess import check_call
projRoot = os.path.dirname(os.path.realpath(__file__)) + '/' + '../../../'


def get_original_dirs(dataset_name):
    uncompressed_dir = projRoot + 'remote_fastdata/omnidata/%s/' % (
        'ta' if dataset_name == 'taskonomy' else 'om'
    )
    subsetNameCandidateList = sorted(os.listdir(uncompressed_dir))
    record = []
    for subset_name in subsetNameCandidateList:
        if os.path.isdir(uncompressed_dir + '%s/%s/' % (subset_name, dataset_name)):
            for scene_name in sorted(os.listdir(
                    uncompressed_dir + '%s/%s/' % (subset_name, dataset_name))):
                record.append({
                    'subset_name': subset_name,
                    'dataset_name': dataset_name,
                    'scene_name': scene_name,
                })
    return record


def main():
    target_dir = '/home/zhusz/local/r/remote_fastdata/omnidata_tools/'
    mkdir_full(target_dir)

    record = []
    for dataset_name in [
        'replica', 'taskonomy', 'blended_mvg', 'clevr_complex', 'clevr_simple',
        'hm3d', 'hypersim', 'replica_gso'
    ]:
        record += get_original_dirs(dataset_name)

    for i, r in enumerate(record):
        if i % 100 == 0:
            print('Working in progress: %d / %d' % (i, len(record)))
        dataset_name = r['dataset_name']
        scene_name = r['scene_name']
        subset_name = r['subset_name']
        # if dataset_name == 'taskonomy' and subset_name == 'mask_valid':  # supplementary links
        if True:
            uncompressed_dir = '../../../omnidata/%s/' % (
                'ta' if dataset_name == 'taskonomy' else 'om'
            )
            mkdir_full(target_dir + '%s/%s/' % (dataset_name, scene_name))
            if not os.path.islink(target_dir + '%s/%s/%s' % (dataset_name, scene_name, subset_name)):
                command_list = [
                    'ln', '-s',
                    uncompressed_dir + '%s/%s/%s' % (subset_name, dataset_name, scene_name),
                    target_dir + '%s/%s/%s' % (dataset_name, scene_name, subset_name),
                ]
                check_call(command_list)

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


def mainTaskonomyMaskValid():
    target_dir = '/home/zhusz/local/r/remote_fastdata/omnidata_tools/'
    mkdir_full(target_dir)
    dataset_name = 'taskonomy'
    subset_name = 'mask_valid'
    uncompressed_dir = '../../../omnidata/ta/'  # '/home/zhusz/omnidata/ta/'

    sceneNameList = sorted(os.listdir(projRoot + 'remote_fastdata/omnidata/ta/' + '%s/%s/' % (subset_name, dataset_name)))

    for scene_name in sceneNameList:
        mkdir_full(target_dir + '%s/%s/' % (dataset_name, scene_name))
        if not os.path.islink(target_dir + '%s/%s/%s' % (dataset_name, scene_name, subset_name)):
            command_list = [
                'ln', '-s',
                uncompressed_dir + '%s/%s/%s' % (subset_name, dataset_name, scene_name),
                target_dir + '%s/%s/%s' % (dataset_name, scene_name, subset_name),
            ]
            check_call(command_list)


if __name__ == '__main__':
    main()
