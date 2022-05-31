% Further Code Release for ICLR-22 work
% 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
% Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
% Released on 05/30/2022.
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.

% Produce the tonemap to ldr with the help of Matlab
clear;

dataset = 'openroomsRendering';
projRoot = [pwd '/' '../../../'];
A1_house = load([projRoot 'v/A/openrooms/A1_order.mat']);
m_house = A1_house.m;
sceneTagList_house = A1_house.sceneTagList;
xmlTagList_house = A1_house.xmlTagList;

A1 = load([projRoot 'v/A/openroomsRendering/A1_order1.mat']);
m = A1.m;
houseIDList = A1.houseIDList;
viewTagList = A1.viewTagList;

hdr_root = [projRoot 'remote_fastdata/openrooms/Image/'];
ldr_root = [projRoot 'remote_fastdata/openrooms/ldrImage/'];
assert(isdir(hdr_root));
mkdir(ldr_root);

for j_matlab = 1:m
    j_python = j_matlab - 1;
    disp(sprintf('Processing script1_tonemap for %s: %d / %d', dataset, j_matlab, m));

    houseID = int32(houseIDList(j_matlab));
    viewTag = int32(viewTagList(j_matlab));

    xmlTag = xmlTagList_house{houseID + 1};
    sceneTag = sceneTagList_house{houseID + 1};

    input_file_name = [hdr_root sprintf('main_%s/%s/im_%d.hdr', xmlTag, sceneTag, viewTag)];
    im = hdrread(input_file_name);
    rgb = tonemap(im);

    mkdir([ldr_root sprintf('main_%s/%s/', xmlTag, sceneTag)]);
    output_file_name = [ldr_root sprintf('main_%s/%s/im_%d.png', xmlTag, sceneTag, viewTag)];
    imwrite(rgb, output_file_name);
end

