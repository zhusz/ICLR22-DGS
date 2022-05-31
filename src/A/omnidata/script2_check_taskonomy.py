# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (tfconda)
import os
import numpy as np
from scipy.stats import mode
from subprocess import check_call


def main():
    # define dirs
    folder_name = 'downloads'
    compress_dir = '/home/zhusz/omnidata/%s/compressed/' % folder_name
    uncompress_dir = '/home/zhusz/omnidata/%s/uncompressed/' % folder_name
    tag_dir = '/home/zhusz/omnidata/%s/tag/' % folder_name

    with open('links_taskonomy.txt', 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    tar_lines = [line for line in lines if line.endswith('.tar')]
    meta_lines = [line for line in lines if not line.endswith('.tar')]

    subsetNameList = []
    sceneNameList = []

    for i, line in enumerate(tar_lines):
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

        if subset_name not in subsetNameList:
            subsetNameList.append(subset_name)
        if scene_name not in sceneNameList:
            sceneNameList.append(scene_name)

    dataset_name = 'taskonomy'
    m_sceneName = len(sceneNameList)
    m_subsetName = len(subsetNameList)
    numFiles = -np.ones((m_sceneName, m_subsetName), dtype=np.int32)
    for j_sceneName, scene_name in enumerate(sceneNameList):
        if j_sceneName % 10 == 0:
            print('Processing %d / %d' % (j_sceneName, m_sceneName))
        for j_subsetName, subset_name in enumerate(subsetNameList):
            tagFileName = tag_dir + '%s-%s-%s.pkl' % (subset_name, dataset_name, scene_name)
            if os.path.isfile(tagFileName):
                numFiles[j_sceneName, j_subsetName] = int(len(os.listdir(
                    uncompress_dir + '%s/%s/%s/' % (subset_name, dataset_name, scene_name)
                )))
    tmp = mode(numFiles, axis=1)
    totFiles = tmp[0][:, 0]
    totCount = tmp[1][:, 0]
    ww = np.where((numFiles - totFiles[:, None] < 0) |
                  (totCount[:, None] == 1) |
                  (totFiles[:, None] == 0))
    indNotComplete = np.stack([ww[0], ww[1]], 1)

    # TODO: filter out indNotComplete[:, 1] == 8 --> they are natually not equal
    flagIndNot8 = (indNotComplete[:, 1] != 8)  #  & (indNotComplete[:, 1] != 13)

    # TODO: filter out numFiles == -1  --> they do not have the tag file generated yet.
    flagIndHaveTagFile = np.zeros((indNotComplete.shape[0], ), dtype=bool)
    for j in range(indNotComplete.shape[0]):
        if numFiles[indNotComplete[j, 0], indNotComplete[j, 1]] >= 0:
            flagIndHaveTagFile[j] = True

    flag = flagIndNot8 & flagIndHaveTagFile
    indNotCompleteFinal = indNotComplete[flag, :]

    import ipdb
    ipdb.set_trace()
    print(1 + 1)
    '''
    for j_scene, j_subset in indNotCompleteFinal:
        subset_name = subsetNameList[j_subset]
        dataset_name = 'taskonomy'
        scene_name = sceneNameList[j_scene]
        tagFileName = tag_dir + '%s-%s-%s.pkl' % (subset_name, dataset_name, scene_name)
        if os.path.isfile(tagFileName):
            check_call(['rm', tagFileName])
    '''

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()
