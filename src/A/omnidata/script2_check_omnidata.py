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
    folder_name = 'd2'
    compress_dir = '/home/zhusz/omnidata/%s/compressed/' % folder_name
    uncompress_dir = '/home/zhusz/omnidata/%s/uncompressed/' % folder_name
    tag_dir = '/home/zhusz/omnidata/%s/tag/' % folder_name

    with open('links_omnidata.txt', 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    tar_lines = [line for line in lines if line.endswith('.tar')]
    meta_lines = [line for line in lines if not line.endswith('.tar')]

    # 'blended_mvg', 'clevr_complex', 'clevr_simple', 'replica_gso', 'replica'
    # 'hm3d': only 00265-depth_euclidean has problem (the given tar is not complete)
    # for dataset_name in ['hypersim']:
    # for dataset_name in ['blended_mvg', 'clevr_complex', 'clevr_simple', 'replica_gso', 'replica', 'hm3d', 'hypersim']:
    for dataset_name in ['hypersim', 'hm3d']:  # only these two guys have corrupted files

        subsetNameList = []
        sceneNameList = []
        for i, line in enumerate(tar_lines):
            flag_omnidata = 'omnidata' in line
            flag_taskonomy = 'taskonomy' in line
            assert (not flag_taskonomy) and flag_omnidata
            fn = os.path.basename(line)
            assert fn.endswith('.tar')

            ss = fn.split('-')
            assert len(ss) >= 3, (ss, fn)
            subset_name = ss[0]

            # dataset_name = ss[1]
            if dataset_name == ss[1]:
                if dataset_name in ['hm3d', 'hypersim', 'replica_gso']:
                    assert len(ss) == 4
                    if dataset_name == 'hm3d':
                        assert len(ss[2]) == 5
                        assert len(ss[3]) == 11 + 4
                    scene_name = ss[2] + '-' + ss[3][:-4]
                else:
                    assert len(ss) == 3, (ss, fn)
                    scene_name = ss[2][:-4]

                if subset_name not in subsetNameList:
                    subsetNameList.append(subset_name)
                if scene_name not in sceneNameList:
                    sceneNameList.append(scene_name)

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
        if dataset_name == 'hypersim':  # actually the same
            tmp = mode(numFiles, axis=1)
            totFiles = tmp[0][:, 0]
            totCount = tmp[1][:, 0]
            ww = np.where((numFiles - totFiles[:, None] < 0) |
                          (totCount[:, None] == 1) |
                          (totFiles[:, None] == 0))
            indNotComplete = np.stack([ww[0], ww[1]], 1)
        else:
            tmp = mode(numFiles, axis=1)
            totFiles = tmp[0][:, 0]
            totCount = tmp[1][:, 0]
            ww = np.where((numFiles - totFiles[:, None] < 0) |
                          (totCount[:, None] == 1) |
                          (totFiles[:, None] == 0))
            indNotComplete = np.stack([ww[0], ww[1]], 1)

        indNotCompleteFinal = indNotComplete
        if dataset_name == 'hm3d':
            indNotCompleteFinal = indNotCompleteFinal[indNotCompleteFinal[:, 1] != 4, :]
        elif dataset_name == 'replica':
            indNotCompleteFinal = indNotCompleteFinal[indNotCompleteFinal[:, 1] != 8, :]
        elif dataset_name == 'hypersim':
            # filter out -1
            flagIndHaveTagFile = np.zeros((indNotCompleteFinal.shape[0],), dtype=bool)
            for j in range(indNotComplete.shape[0]):
                if numFiles[indNotComplete[j, 0], indNotComplete[j, 1]] >= 0:
                    flagIndHaveTagFile[j] = True
            indNotCompleteFinal = indNotCompleteFinal[flagIndHaveTagFile, :]

        '''
        for j_scene, j_subset in indNotCompleteFinal:
            tagFileName = tag_dir + '%s-%s-%s.pkl' % \
                          (subsetNameList[j_subset], dataset_name, sceneNameList[j_scene])
            if os.path.isfile(tagFileName):
                check_call(['rm', tagFileName])
        '''

        import ipdb
        ipdb.set_trace()
        print(1 + 1)

        # compile corrupted file list
        with open('./corrupted_list_%s.txt' % dataset_name, 'w') as f:
            for scene_id, subset_id in zip(indNotComplete[:, 0], indNotComplete[:, 1]):
                scene_name = sceneNameList[scene_id]
                subset_name = subsetNameList[subset_id]
                if subset_name not in ['nonfixated', 'nonfixated_matches']:
                    f.write('%s %s %s\n' % (dataset_name, scene_name, subset_name))

        del subsetNameList, sceneNameList

    import ipdb
    ipdb.set_trace()
    print(1 + 1)


if __name__ == '__main__':
    main()
