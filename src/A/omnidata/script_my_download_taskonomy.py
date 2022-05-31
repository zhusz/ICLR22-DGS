# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# bash for parallel :
# for g in {0..20}; do J1=$((g*500)) J2=$(((g+1)*500)) python script_my_download_taskonomy.py & done
# for g in {0..96}; do J1=$((g*100)) J2=$(((g+1)*100)) python script_my_download_taskonomy.py & done
from subprocess import check_call
# import urllib.request
import os
import pickle

import ipdb


def main():
    folder_name = 'downloads'
    compress_dir = '/home/zhusz/omnidata/%s/compressed/' % folder_name
    uncompress_dir = '/home/zhusz/omnidata/%s/uncompressed/' % folder_name
    tag_dir = '/home/zhusz/omnidata/%s/tag/' % folder_name
    for x in [compress_dir, uncompress_dir, tag_dir]:
        if not os.path.isdir(x):
            os.mkdir(x)

    # url = 'https://datasets.epfl.ch/taskonomy/links.txt'
    # file = urllib.request.urlopen(url)
    # lines = [line.decode('utf-8').strip() for line in file]
    # lines = [line for line in lines if line.endswith('.tar')]
    with open('links_taskonomy.txt', 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line.endswith('.tar')]

    m = len(lines)  # 9648
    for i, line in enumerate(lines):
        assert line.endswith('.tar'), (i, line)
        assert ' ' not in line, (i, line)

    j1 = int(os.environ['J1'])
    j2 = int(os.environ['J2'])
    assert 0 <= j1 < j2
    indChosen = list(range(j1, j2))

    for j in indChosen:
        line = lines[j]
        flag_omnidata = 'omnidata' in line
        flag_taskonomy = 'taskonomy' in line
        assert flag_taskonomy and (not flag_omnidata)
        fn = os.path.basename(line)
        assert fn.endswith('.tar')

        assert '_' in fn
        ss = fn.split('_')
        subset_name = '_'.join(ss[1:])[:-4]  # remove '.tar'
        dataset_name = 'taskonomy'
        scene_name = ss[0]

        tagFileName = tag_dir + '%s-%s-%s.pkl' % (subset_name, dataset_name, scene_name)

        if os.path.isfile(tagFileName):
            print('Skipping %d (j1 = %d, j2 = %d' % (j, j1, j2))
        else:
            print('script_my_download of omnidata: %d (j1 = %d, j2 = %d) - %s' %
                  (j, j1, j2, lines[j]))
            try:

                wget_command_list = ['wget', '-P', compress_dir, line, '-q', '-o', '/dev/null']
                check_call(wget_command_list)  # no timeout in case too large file

                # tar -xf
                direct_dir = uncompress_dir + '%s/%s/%s' % \
                             (subset_name, dataset_name, scene_name)
                if not os.path.isdir(direct_dir):
                    os.makedirs(direct_dir, exist_ok=True)
                tar_command_list = [
                    'tar', '-xf', compress_dir + fn, '-C', direct_dir, '--strip-components=1'
                ]
                check_call(tar_command_list)

                # remove the compressed
                rm_command_list = ['rm', compress_dir + fn]
                check_call(rm_command_list)

                with open(tagFileName, 'wb') as f:
                    pickle.dump({
                        'finishTag': 1,
                    }, f)
            except:
                pass


if __name__ == '__main__':
    raise ValueError('No longer used.')
    main()
