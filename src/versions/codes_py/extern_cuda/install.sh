#!/usr/bin/env bash

pip install Cython
for i in $(ls | fgrep -v install | fgrep -v clear);
do
    echo "Working on $i"
    cd $i
    rm -r build
    INSTALLATIONS_DIR=$1 CUDA_VISIBLE_DEVICES=$2 python setup.py install
    cd ..
done
