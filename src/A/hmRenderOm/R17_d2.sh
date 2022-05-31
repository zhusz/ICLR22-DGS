# Further Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 05/30/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 0-480W
for g in {0..5}
do
    for t in {0..7}
    do
        CUDA_VISIBLE_DEVICES=$g J1=$((100000*(g*8+t))) J2=$((100000*(g*8+t+1))) PYOPENGL_PLATFORM=osmesa python R17_faceFlag.py &
    done
done