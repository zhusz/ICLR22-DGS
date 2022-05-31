#!/usr/bin/env bash


for i in $(ls | fgrep -v install | fgrep -v clear);
do
    echo "Working on $i"
    cd $i
    rm -r build
    rm -r dist
    rm -r *.egg-info
    cd ..
done
