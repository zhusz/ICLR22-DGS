% Further Code Release for ICLR-22 work
% 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
% Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
% Released on 05/30/2022.
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.

% (matlab)
clear;

projRoot = [pwd '/' '../../../'];
A1_house = load([projRoot 'v/A/openrooms/A1_order.mat']);
m = A1_house.m;
sceneList_house = A1_house.sceneList;
xmlTagList_house = A1_house.xmlTagList;

hdr_root = [projRoot 'remote_fastdata/openrooms/Image/'];
target_root = [projRoot 'cache/dataset/openrooms/Image/'];
assert(isdir([projRoot 'cache/']));
mkdir(target_root);

for j_house = 1:5
    sceneTag = sceneList_house{j_house};
    xmlTag = xmlTagList_house{j_house};

    % only check view 1
    viewID = 1;

    input_file_name = [hdr_root sprintf('main_%s/%s/im_%d.hdr', xmlTag, sceneTag, viewID)];

    im = hdrread(input_file_name);
    rgb = tonemap(im);

    imwrite(rgb, [target_root sprintf('rgb_%s_%s_%d_matlabDefault.png', xmlTag, sceneTag, viewID)]);
    copyfile(input_file_name, [target_root sprintf('rgb_%s_%s_%d.hdr', xmlTag, sceneTag, viewID)]);

    lab = rgb2lab(rgb);
    lightness_pool = {[0, 0.4], [0.6, 1], [0.4, 0.6], [0, 1]};
    saturation_pool = [0.2, 1, 5];
    for l = 1:4
        for s = 1:3
            lightness = lightness_pool{l};
            low = single(lightness(1));
            high = single(lightness(2));
            saturation = saturation_pool(s);
            lab_now = lab;
            lab_now(:, :, 1) = (lab_now(:, :, 1) / 100. * (high - low) + low) * 100.;
            lab_now(:, :, 2:3) = lab_now(:, :, 2:3) * saturation;
            rgb_now = lab2rgb(lab_now);
            imwrite(rgb_now, [target_root sprintf('rgb_%s_%s_%d_l_%.1f_h_%.1f_s_%.1f.png', xmlTag, sceneTag, viewID, low, high, saturation)]);
        end
    end
end

% Play conclusion:
% Tried matlab version of the tone mapping. File size indeed decreased (x3, png).
% But the result does not emphasize clear color (contrast).
% Various way of tone mapping itself is a good way of data augmentation.

% So, dump the matlab version of the tone mapping, and also prepare an online tone-mapping python algorithm.

% Note: After installing Adobe PhotoShop, the preview vis of the hdr file is just what PhotoShop will show.


% If python can output an equivalent of tone mapping result, then matlab dump would be unnecessary


% Final Strategy after played with Python side:
% Use matlab to generate png, only one png per image.
% You can augment in the LAB space on the fly (lightness and saturation).
% For details, please take a look at the tonemap.m file.
% This pixel level data augmentation strategy basically can be applied to almost all the
% other training images.

% If you do not have matlab, you can still impmlement it on your own,
% first map the hdr into log2 space, and then rgb2lab, and then use histogram harmonization on the L channel,
% pay attention to the public doc of the adapthisteq / imadjust functions
% and finally, map the L within (0, 100) to your elevated range, and a/b channel times by saturation (for colorization enhancement).
