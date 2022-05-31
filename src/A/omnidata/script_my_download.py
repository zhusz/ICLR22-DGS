# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# bash for parallel :
# for g in {0..18}; do J1=$((g*2000)) J2=$(((g+1)*2000)) python script_my_download.py & done
from subprocess import check_call
import os
import pickle


def main():
    raise NotImplementedError("No longer used.")
    folder_name = 'downloads'
    compress_dir = '/home/zhusz/omnidata/%s/compressed/' % folder_name
    uncompress_dir = '/home/zhusz/omnidata/%s/uncompressed/' % folder_name
    tag_dir = '/home/zhusz/omnidata/%s/tag/' % folder_name
    for x in [compress_dir, uncompress_dir, tag_dir]:
        if not os.path.isdir(x):
            os.mkdir(x)

    with open('./total.txt', 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    m = len(lines)  # 36121  # 10 minutes for 134. Roughly needs 44 hours in total.
    for i, line in enumerate(lines):
        assert line.endswith('.tar'), (i, line)
        assert ' ' not in line, (i, line)

    j1 = int(os.environ['J1'])
    j2 = int(os.environ['J2'])
    assert 0 <= j1 < j2
    indChosen = list(range(j1, j2))

    this_root = os.path.dirname(os.path.realpath(__file__)) + '/'
    for j in indChosen:
        if os.path.isfile(tag_dir + '%08d.pkl' % j):
            print('Skipping %d (j1 = %d, j2 = %d' % (j, j1, j2))
        else:
            print('script_my_download of omnidata: %d (j1 = %d, j2 = %d)' %
                  (j, j1, j2))
            try:
                line = lines[j]

                flag_omnidata = 'omnidata' in line
                flag_taskonomy = 'taskonomy' in line
                assert int(flag_omnidata) + int(flag_taskonomy) == 1
                if flag_taskonomy:
                    continue  # not handling that in this script.

                fn = os.path.basename(line)
                assert fn.endswith('.tar')
                wget_command_list = ['wget', '-P', compress_dir, line, '-q', '-o', '/dev/null']
                check_call(wget_command_list)  # no timeout in case too large file
                if flag_omnidata:
                    tar_command_list = [
                        'tar', '-xf', compress_dir + fn, '-C', uncompress_dir, '--strip-components=0']
                else:
                    pass  # skip taskonomy - this one use the original aria2 to download.
                    continue
                    # assert '_' in fn
                    # ss = fn.split('_')
                    # scene_name = ss[0]
                    # subset_name = ss[1][:-4]  # remove '.tar'
                    # direct_dir = uncompress_dir + '%s/%s/%s/' % (subset_name, 'taskonomy', scene_name)
                    # os.makedirs(direct_dir, exist_ok=True)
                    # tar_command_list = [
                    #     'tar', '-xf', compress_dir + fn, '-C', direct_dir
                    # ]
                check_call(tar_command_list)
                rm_command_list = ['rm', compress_dir + fn]
                check_call(rm_command_list)
                # check_call(['ls'])
                # rm_wget_log_command_list = ['rm', this_root + 'wget-log*']
                # check_call(rm_wget_log_command_list)
                with open(tag_dir + '%08d.pkl' % j, 'wb') as f:
                    pickle.dump({
                        'finishTag': 1,
                    }, f)
            except:
                pass


if __name__ == '__main__':
    main()
