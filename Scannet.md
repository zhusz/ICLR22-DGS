# Scannet Experiments

## Pre-Installation Steps - Storage

Make sure that the Pre-Installations Steps have been conducted following 
[this](https://github.com/zhusz/ICLR22-DGS/blob/master/ShapeNet.md#pre-installation-steps---storage).

## Installations for the Scannet Experiments

Our code base relies on PyTorch, PyTorch3D, pyrender and pyopengl. In particular, our own implementation is compatible to most of recent versions of the software above, and hence please just make sure the software packages above can co-exist with each other. We have tested our codes on the latest version of all these packages, in particular, with PyTorch 1.10.1, PyTorch3D 0.6.1, CUDA 11, on Nvidia Ampere devices, while we believe earlier version would also be compatible. Please follow the following steps to install.

[New] Our recent experience indicated that our codes can run with PyTorch 1.11, PyTorch3D 0.6.2, CUDA 11 on Ampere Devices.

1. Create a new conda environment. (You can also use virtualenv or other equivalents)

    ```
    conda create -n dgs_scannet python=3
    conda activate dgs_scannet
    ```
   
   All the following operations are presumed to be operated under the `dgs_scannet` python environment.

2. Install PyTorch.

    ```
    conda install pytorch torchvision cudatoolkit -c pytorch
    ```
   
3. Install ninja.

    ```
    sudo apt update
    sudo apt install ninja-build
    pip install ninja
    ```
   
4. Build PyTorch3D from source.

    ```
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    cd $INSTALLATIONS_DIR
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d
    pip install .
    cd ..
    rm -rf pytorch3d
    ```

5. Install pyrender and pyopengl (follwing [this](https://pyrender.readthedocs.io/en/latest/install/index.html) instruction).

   ```
   sudo apt update
   wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
   sudo dpkg -i ./mesa_18.3.3-0.deb || true
   sudo apt install -f
   pip install pyrender
   git clone https://github.com/mmatl/pyopengl.git
   pip install ./pyopengl
   ```
   
   Everytime when running, you need to set this environment variable:

   ```
   PYOPENGL_PLATFORM=osmesa
   ```
   
8. Install other dependencies.

   ```
   cd $PROJ_DIR
   pip install -r environments/t11/requirements.txt
   ```
   
7. Prepare Eigen, and then compile our cuda codes.

   ```
   cd $INSTALLATIONS_DIR
   mkdir eigen
   cd eigen
   wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
   tar -xjf eigen-3.4.0.tar.bz2
   mv eigen-3.4.0 eigen3
   rm eigen-3.4.0.tar.bz2
   cd $PROJ_DIR/src/versions/codes_py/extern_cuda/
   bash install.sh $INSTALLATIONS_DIR
   cd $INSTALLATIONS_DIR
   rm -rf eigen
   ```

## Running the Demo

1. First please make sure you have several test images available. Test images can be arbitrary 
as long as they are indoor scenes. Please allow failure cases for future improvements. 
Our model will crop out the maximum area of central crop whose aspect ratio is the same as in
our training (192x256), and our predictions are only based on the cropped area.

   Download the unseen scene test images in our paper via

   ```
   cd $PROJ_DIR/v/A/freedemo1/
   wget https://www.dropbox.com/s/g9yblq66g8ru5kw/freedemo1.tar.gz
   tar -xzf freedemo1.tar.gz
   rm freedemo1.tar.gz
   ```

   Make sure the test images (including your own) are placed in the right directory:

   ```
   |- $PROJ_DIR
      |- v
         |- A
            |- freedemo1
               |- *.png
               |- *.jpg
               |- ...
   ```
   
4. Then download our new pretrained model from ([Encoder](TODO), [Decoder](TODO))*.

   Alternatively download our older submission-time model from
   ([Encoder](https://www.dropbox.com/s/8twmf10nto25zdl/encoder_net_latest.pth), [Decoder](https://www.dropbox.com/s/s1botf6b9g8r9v5/decoder_net_latest.pth)) &dagger;.

   Place the models under the following directories (of their linked locations).

   Note: our new model* has been trained with new techniques which is not obtained via the released training code. Our submission-time model&dagger; is trained by the released training code.
   
   ```
   |- $PROJ_DIR
      |- v
         |- P
            |- Pscannet
               |- D0dgs
                  |- S101
                     |- Rrelease
                        |- models
                           |- encoder_net_latest.pth
                           |- decoder_net_latest.pth
                           |- ...
   ```
  
5. Run the following commands to run the model predictions.
   
   ```
   cd $PROJ_DIR/src/B/
   
   # Try our new pretrained model
   MC=DGS-New DS=freedemo1 PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES=0 python -m Bpscannet.run
   
   # Try our older submission-time model (requiring post-processing)
   MC=DGS-Submission-Time DS=freedemo1 PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES=0 python -m Bpscannet.run
   ```

6. Retrieve your predicted results via

   ```
   DS=freedemo1 python -m Bpscannet.csvGeometry3DEntry.dumpHtmlForPrepickGeometry3D
   ```

7. View your results using a browser of the following webpage

   ```
   google-chrome $PROJ_DIR/cache/B_Bpscannet/freedemo1_dumpHtmlForPrepick/html_freedemo1/html_000000/index.html
   ```
   
## Downloading Datasets

Our model requires the [Scannet-v2](http://www.scan-net.org/changelog#scannet-v2-2018-06-11) 
dataset. Please obtain the data according to [this](https://github.com/ScanNet/ScanNet#scannet-data).

The directory should look like this:

```
|- $PROJ_DIR/
   |- remote_fastdata/
      |- scannet/
         |- scans/  # The downloaded scannet trainval set
            |- scene0000_00/
               |- *.ply
               |- *.json
               |- *.sens
               |- ...
            |- sceneXXXX_XX/
            |- ...
         |- scans_test/  # The downlaoded scannet test set
            |- scene0707_00/
               |- *.ply
               |- *.sens
```
<!--
      |- scannet_cache/  # The generated color / depth / extrinsic poses / intrinsic poses
         |- scans/
            |- color/
               |- *.jpg
            |- depth/
               |- *.png
            |- pose/
               |- *.txt
            |- intrinsic/  # (4 .txt files in it)
         |- scans_test/
            |- color/
               |- *.jpg
            |- depth/
               |- *.png
            |- pose/
               |- *.txt
            |- intrinsic/  # (4 .txt files in it)
-->

## Downloading Pre-pretrained Weights as Model Initialization
Obtain the pre-pretrained weights `res101.pth` from [here](https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31) (Their Step 1)
and put according to the following structure

```
|- $PROJ_DIR/
    |- v_external_codes/
        |- AdelaiDepth/
            |- LeRes/
                |- res101.pth
```

## Intermediate Training Data Generation

1. Download the meta data.

   ```
   cd $PROJ_DIR/v/Z/
   wget https://www.dropbox.com/s/0yyojpxsjxmb5ai/Z.tar.gz
   tar -xzf Z.tar.gz
   rm Z.tar.gz
   ```

1. Run the following scripts to obtain intermediate data related to Scannet.

   ```
   cd $PROJ_DIR/src/A/scannet
   python A1_m.py
   python A01_ind.py
   for g in {0..95}; do J1=$((17*g)) J2=$((17*(g+1))) python R17.py & done
   ```

2. Extract the color / depth / extrinsic poses / intrinsic poses via running the following script:

   ``` 
   cd $PROJ_DIR/src/A/scannet/
   MP=0 python cache_color_depth_pose_intrinsics.py
   ```
   
   Note this step might cost considerable amount of time. To accelerate with multi-threads, simply 
running 

   ```
   MP=32 python cache_color_depth_pose_intrinsics.py
   ```

3. Run the following scripts to obtain other important intermediate training data.

   ```
   cd $PROJ_DIR/src/A/scannetGivenRender
   python A1_m.py
   python A1b_supplement.py
   python A01_ind.py
   for g in {0..26}; do J1=$((g*100000)) J2=$(((g+1)*100000)) CUDA_VISIBLE_DEVICES=$g PYOPENGL_PLATFORM=osmesa python R1.py & done  
   for g in {0..26}; do J1=$((g*100000)) J2=$(((g+1)*100000)) CUDA_VISIBLE_DEVICES=$g PYOPENGL_PLATFORM=osmesa python R17.py & done  
   ```
   
   Once again, the last line costs consideratble amount of time even if paralleled.

## Training

Using the following commands to train our DGS model on Scannet.

```
cd $PROJ_DIR/src/P/
D=D0dgs S=S101 R=R0 CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=osmesa python -m Pscannet.run
```

To run with multiple GPUs (4 GPUs for example), use the following commands.

```
cd $PROJ_DIR/src/P/
D=D0dgs S=S101 R=R0 CUDA_VISIBLE_DEVICES=0,1,2,3 PYOPENGL_PLATFORM=osmesa python -m \
    torch.distributed.launch --master_port=12345 --nproc_per_node=4 \
    --use_env -m Pscannet.run
```

Convergence requires roughly 100K-200K iterations. Please refer to [this part](https://github.com/zhusz/ICLR22-DGS/blob/master/Scannet.md) (Step 2)
for pretrained weights.

## Testing

At the point of submission, we used the then-version of benchmarking. 
Use the following command to obtain the that version of benchmarking.

```
cd $PROJ_DIR/src/B/
MC=DGS-Submission-Time-No-PostProcess DS=submissionTimeScannetOfficialTestSplit10 CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=osmesa python -m Bpscannet.run
```

To Accelerate, for example if 10 GPUs are available

```
cd $PROJ_DIR/src/B/
for g in {0..9}; do J1=$((100*g)) J2=$((100*(g+1))) MC=DGS-Submission-Time-No-PostProcess DS=submissionTimeScannetOfficialTestSplit10 CUDA_VISIBLE_DEVICES=$g PYOPENGL_PLATFORM=osmesa python -m Bpscannet.run & done
# Run again the single-thread to read from cache and tabulate the performance
MC=DGS-Submission-Time-No-PostProcess DS=submissionTimeScannetOfficialTestSplit10 CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=osmesa python -m Bpscannet.run 
```

The following benchmarking result would show.

```
+------------------------------------+--------+----------+------------+---------+-----------+-------+-----------+------------+----------+---------+------------+-------+-------+-------+-------------+-----------+-------------+---------------+------------+--------------+----------+--------------+---------------+-------------+------------+---------------+----------+----------+----------+----------------+
|               method               | FitAcc | FitCompl | FitChamfer | FitPrec | FitRecall | FitF1 | FitAbsRel | FitAbsDiff | FitSqRel | FitRMSE | FitLogRMSE | FitR1 | FitR2 | FitR3 | FitComplete | MetricAcc | MetricCompl | MetricChamfer | MetricPrec | MetricRecall | MetricF1 | MetricAbsRel | MetricAbsDiff | MetricSqRel | MetricRMSE | MetricLogRMSE | MetricR1 | MetricR2 | MetricR3 | MetricComplete |
+------------------------------------+--------+----------+------------+---------+-----------+-------+-----------+------------+----------+---------+------------+-------+-------+-------+-------------+-----------+-------------+---------------+------------+--------------+----------+--------------+---------------+-------------+------------+---------------+----------+----------+----------+----------------+
| DGS-Submission-Time-No-PostProcess |  14.3  |   11.5   |    12.9    |   39.7  |    49.4   |  41.6 |    8.9    |    14.7    |   4.1    |   22.7  |    12.7    |  91.1 |  97.5 |  99.3 |     76.5    |    15.2   |     11.5    |      13.4     |    31.6    |     43.3     |   35.2   |     13.7     |      22.6     |     7.4     |    30.8    |      17.4     |   83.1   |   94.7   |   98.0   |      76.5      |
+------------------------------------+--------+----------+------------+---------+-----------+-------+-----------+------------+----------+---------+------------+-------+-------+-------+-------------+-----------+-------------+---------------+------------+--------------+----------+--------------+---------------+-------------+------------+---------------+----------+----------+----------+----------------+
```

Notably, during the rebuttal process, we noticed that this version might result
in problems on a small number of test cases, and in particular, some test cases
demonstrates only a flat wall in the space, leading to the problems of "no 
predicted surface presented in the space of interest" (now we define the space 
of interest during evaluation to be the convex hull of the depth point cloud).
Hence, we modify the evaluation routine from [delinate-first-then-fit](https://github.com/zhusz/ICLR22-DGS/blob/master/src/B/Bpscannet/csvGeometry3DEntry/benchmarkingGeometry3DSubmissionTimeScannet.py#L182) 
to [fit-first-then-delineate](https://github.com/zhusz/ICLR22-DGS/blob/master/src/B/Bpscannet/csvGeometry3DEntry/benchmarkingGeometry3DScannet.py#L159). As a result, our bug-fixed benchmarking 
would yield the following result if we run this command, and we hope future
comparisons are conducted with this new benchmarking routine.

```
cd $PROJ_DIR/src/B/
MC=DGS-Submission-Time-No-PostProcess DS=scannetOfficialTestSplit10 CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=osmesa python -m Bpscannet.run
```

```
+------------------------------------+--------+----------+------------+---------+-----------+-------+-----------+------------+----------+---------+------------+-------+-------+-------+-------------+-----------+-------------+---------------+------------+--------------+----------+--------------+---------------+-------------+------------+---------------+----------+----------+----------+----------------+
|               method               | FitAcc | FitCompl | FitChamfer | FitPrec | FitRecall | FitF1 | FitAbsRel | FitAbsDiff | FitSqRel | FitRMSE | FitLogRMSE | FitR1 | FitR2 | FitR3 | FitComplete | MetricAcc | MetricCompl | MetricChamfer | MetricPrec | MetricRecall | MetricF1 | MetricAbsRel | MetricAbsDiff | MetricSqRel | MetricRMSE | MetricLogRMSE | MetricR1 | MetricR2 | MetricR3 | MetricComplete |
+------------------------------------+--------+----------+------------+---------+-----------+-------+-----------+------------+----------+---------+------------+-------+-------+-------+-------------+-----------+-------------+---------------+------------+--------------+----------+--------------+---------------+-------------+------------+---------------+----------+----------+----------+----------------+
| DGS-Submission-Time-No-PostProcess |  13.8  |   9.2    |    11.5    |   40.4  |    53.3   |  44.2 |    9.0    |    14.9    |   4.1    |   22.9  |    12.8    |  90.9 |  97.6 |  99.3 |     90.2    |    15.7   |     10.4    |      13.0     |    28.9    |     43.9     |   34.1   |     14.2     |      23.4     |     7.5     |    31.5    |      17.8     |   82.4   |   94.5   |   98.0   |      90.2      |
+------------------------------------+--------+----------+------------+---------+-----------+-------+-----------+------------+----------+---------+------------+-------+-------+-------+-------------+-----------+-------------+---------------+------------+--------------+----------+--------------+---------------+-------------+------------+---------------+----------+----------+----------+----------------+
```

<!--
## Misc

https://www.dropbox.com/sh/td36r00y7cd6ano/AAAefukZOCsRRvgZEE6_R4i0a?dl=0

-->
