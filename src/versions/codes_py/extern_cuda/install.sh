# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env bash

pip install Cython
for i in $(ls | fgrep -v install | fgrep -v clear);
do
    echo "Working on $i"
    cd $i
    rm -r build
    INSTALLATIONS_DIR=$1 python setup.py install
    cd ..
done
