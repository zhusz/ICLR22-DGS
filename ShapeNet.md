# ShapeNet Experiments

## Pre-Installation Steps - Storage

1. Define the following two dirs that will be frequently referred to throughout the docs.
They can be at different locations.

   ```
   INSTALLATIONS_DIR
   WORK_DIR
   ```
   
   We also define `PROJ_DIR` to be `$WORK_DIR/ICLR22-DGS`.

   You do not have to (while recommended for conveniency) export these environmental variables or put them into `~/.bashrc`, but instead, remember their locations as they will be frequently referred to throughout the docs.

2. Clone the repository via

   ```
   cd $WORK_DIR
   git clone git@github.com:zhusz/ICLR22-DGS.git
   ```

3. Define three storing spaces with respect to three levels of read/write speed. Construct the link to each of them.
You can of course link them to the same directories (if all of your
storage space demonstrate the same level of io speed).

   ```
   cd $PROJ_DIR
   ln -s [Path of Your Choice] $PROJ_DIR/remote_fastdata  # fastest, e.g. your local SSD
   ln -s [Path of Your Choice] $PROJ_DIR/remote_slowdata  # slower, can be a remote disk
   ln -s [Path of Your Choice] $PROJ_DIR/remote_syncdata  # Just use a remote disk
   ```
   
4. `mkdir` for all the link targets under `$PROJ_DIR/v/` and `$PROJ_DIR/v_external_codes/`. You do not need
to finish this at once at this point, but when prompted with errors in
future steps, remember to create these directories.

## Installations for the ShapeNet Experiments
   
Our code base relies on PyTorch, PyTorch3D, pyrender and pyopengl.
In particular, our own implementation is compatible to most of recent versions of the software above,
and hence please just make sure the software packages above can co-exist with each other.
We have tested our codes on the latest version of all these packages,
in particular, with PyTorch 1.10.1, PyTorch3D 0.6.1,
CUDA 11, on Nvidia Ampere devices,
while we believe newer or older version would also be compatible.
Please follow the following steps to install with the latest versions
of the environment.

[New] Our recent experience indicated that our codes can run with PyTorch 1.11, PyTorch3D 0.6.2, CUDA 11 on
Ampere Devices.

1. Create a new conda environment. (You can also use virtualenv or other equivalents)

    ```
    conda create -n dgs_shapenet python=3
    conda activate dgs_shapenet
    ```
   
   All the following operations are presumed to be operated under the `dgs_shapenet` python environment.

2. Install PyTorch.

    ```
    conda install pytorch torchvision cudatoolkit -c pytorch
    ```
   
3. Install ninja, g++ and others.

    ```
    sudo apt update
    sudo apt install g++-8 ninja-build libboost-container-dev unzip
    ```
   
5. Build PyTorch3D from source.

    ```
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    cd $INSTALLATIONS_DIR
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d
    pip install .
    cd ..
    rm -rf pytorch3d
    ```

6. Install the following packages.

   ```
   pip install google-cloud-storage moderngl Pillow PyOpenGL ninja
   ```
   
   This step is following the [instructions](https://github.com/google-research/corenet/blob/main/requirements.txt) in CoReNet.
   
8. Install other dependencies.

   ```
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
  
## Downloading Datasets

1. Download the ShapeNetV2 dataset, and put the data according to 
the following structure

    ```
    |-$PROJ_DIR/
        |- remote_fastdata/
            |- shapenetcore_v2/
                |- taxonomy.json
                |- 02691156/
                |- ...
    ```
   
2. Download the rendering via 

    ```
    mkdir $PROJ_DIR/remote_fastdata/corenet/data/raw
    cd $PROJ_DIR/remote_fastdata/corenet/
    for n in single; do  
      for s in train val test; do
        wget "https://storage.googleapis.com/gresearch/corenet/${n}.${s}.tar" \
          -O "data/raw/${n}.${s}.tar" 
        tar -xvf "data/raw/${n}.${s}.tar" -C data/ 
      done 
    done
    ```
   
    The resulting storage structure should look like this

    ```
    |- $PROJ_DIR/
        |- remote_fastdata/
            |- corenet/
                |- data/
                    |- single.train/
                    |- single.val/
                    |- single.test/
    ```
   
## Downloading Pre-pretrained Weights as Model Initialization

Obtain the pre-pretrained weights `keras_resnet50_imagenet.cpt` via running [this script](https://github.com/google-research/corenet/blob/main/src/import_resnet50_checkpoint.py),
and put the resulting file under the following structure

```
|- $PROJ_DIR/
   |- v_external_codes/
      |- corenet/
         |- data/
            |- keras_resnet50_imagenet.cpt
```

<!-- Alternatively, obtain the pre-pretrained weights directly from [here](https://www.dropbox.com/s/feu85z8c84t9m89/keras_resnet50_imagenet.cpt). -->
   
## Intermediate Training Data Generation

1. Download the metadata file [here](https://www.dropbox.com/s/e93hl9gh4odq9jr/A1b_order1Supp.pkl)
and put according to the following structure

    ```
    |- $PROJ_DIR/
        |- v/
            |- A/
                |- corenetChoySingleRendering/
                    |- A1b_order1Supp.pkl
    ```

    Or alternatively, run the following scripts to generate this file.

    ```
    cd $PROJ_DIR/src/A/corenetSingleRendering/
    python A1_order1.py
    python A1b_order1Supp.py
    cd $PROJ_DIR/src/A/corenetChoySingleRendering/
    python A1b_main.py
    ```
   
2. Generate the pkl files for all the meshdata from ShapeNet.

    ```
    cd $PROJ_DIR/src/A/corenetChoySingleRendering/
    python script_pklize_shape_obj.py
    ```


## Errata

Initial versions of this code base had a [bug](https://github.com/zhusz/ICLR22-DGS/blob/master/src/P/Pshapenet/visual.py#L57) that led our evaluations to run on the resolution of 64 instead of 128.

<!--
We found an [error](https://github.com/zhusz/ICLR22-DGS/blob/master/src/P/Pshapenet/visual.py#L53) in our old code base
(Btag: `Bpshapenet`, Ptag: `Pshapenet`)
that our whole evaluations were evaluated based on the 64x64x64 voxel grids from the very beginning of the project
(and never switched to 128 throughout the project).
Such error is as a result of the illogical code base design of our old version of the code base that critical
benchmarking parameters are determined inside the [approach section](https://github.com/zhusz/ICLR22-DGS/blob/master/src/P/Pshapenet/visual.py#L53)
rather than the [benchmarking section](https://github.com/zhusz/ICLR22-DGS/blob/master/src/B/Bpshapenet2/csvShapenetEntry/Bpshapenet2_1D128_corenet.py#L124).
We initially set the resolution to be 64 as it runs extremely slowly for the implicit case on the 128 resolution
(try this to test the speed under the extreme case, that takes 9 days to finish benchmarking
`cd $PROJ_DIR/src/B/; MC=Oracle-DISN-DVR-128 DS=corenetSingleOfficialTestSplit CUDA_VISIBLE_DEVICES=0 python -m Bpshapenet2.run`).
-->

We have updated with the new code base (under `src/B/Bpshapenet2` (for benchmarking) and `src/P/Pshapenet2` (for models and training), if you wish to
use the released pretrained weight, please also download accordingly using the same download link, but put under `Pshapenet2`),
in which you can run the evaluation with
both resolutions. It also support multi-GPU. Try running the following to get the 128-resolution result of the DGS model with 2 GPUs:

```
cd $PROJ_DIR/src/B/
for g in {0..1}; do MC=Ours-DGS-128 CUDA_VISIBLE_DEVICES=$g J1=$((949*g)) J2=$((949*(g+1))) python -m Bpshapenet2.run & done
# Run again the single-thread to read from cache and tabulate the performance
MC=Ours-DGS-128 CUDA_VISIBLE_DEVICES=0 python -m Bpshapenet2.run
```

The following should be expected to show:

```
+--------------+---------------+----------+------------+---------+------------+------------+----------+----------+--------+-------------+---------+----------+----------+---------+
|    method    | watercraftIou | rifleIou | displayIou | lampIou | speakerIou | cabinetIou | chairIou | benchIou | carIou | airplaneIou | sofaIou | tableIou | phoneIou | meanIou |
+--------------+---------------+----------+------------+---------+------------+------------+----------+----------+--------+-------------+---------+----------+----------+---------+
| Ours-DGS-128 |      61.1     |   67.5   |    62.7    |   44.2  |    54.8    |    49.6    |   59.5   |   45.4   |  59.4  |     59.9    |   69.8  |   55.1   |   78.0   |   59.0  |
+--------------+---------------+----------+------------+---------+------------+------------+----------+----------+--------+-------------+---------+----------+----------+---------+
```

Note we have not tested the training code under Ptag `Pshapenet2` but we believe it will result in the same training process.
Use the training code under `src/P/Pshapenet` to reproduce our training.

The documentation starting from here on, is our original documentation. 
  
## Training 

Using the following commands to train our DGS model on ShapeNet.

```
cd $PROJ_DIR/src/P/
D=D0corenetdgs S=S2 R=R0 CUDA_VISIBLE_DEVICES=0 python -m Pshapenet.run
```

We also provide the CoReNet baseline as the direct ablation baseline 
(with the only differences at the sampling and the loss).

```
cd $PROJ_DIR/src/P/
D=D0corenet S=S2 R=R0 CUDA_VISIBLE_DEVICES=0 python -m Pshapenet.run
```

Converged performance is listed below.

| P | D | S | Model Description | MeanIOU (&#8593;) | Craft | Rifle | Disp. | Lamp | Speaker | Cabinet | Chair | Bench | Car | Plane | Sofa | Table | Phone | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| Pshapenet | D0corenetdgs | S2 | (Ours) DGS | 62.9 |    63.3     |   70.4   |    65.7    |   49.0  |    61.2    |    57.0    |   62.1   |   50.9   |  70.1  |     62.8    |   70.3  |   56.9   |   78.3  | 
| Pshapenet | D0corenet | S0 | (Oracle) [CoReNet](https://github.com/google-research/corenet) | 61.5 |    62.5     |   65.5   |    63.2    |   48.2  |    63.0    |    56.7    |   60.6   |   48.3   |  73.7  |     58.1    |   69.8  |   55.0   |   75.1    | 

Our model converges roughly in 3M iterations.

Note that in order to serve as the direct ablation baseline,
the CoReNet here works purely on the voxels, and we omit their offset 
``o``. This won't affect the resolution of the reconstructions significantly, 
and the performance above indicated that this version is the same 
as in the [official implementation](https://github.com/google-research/corenet#training-and-evaluating-a-new-model) (h7),
and better than the [original performance](https://github.com/google-research/corenet#models-from-the-paper) (h7).

We also release another pair of the experiments to serve as 
the best model we obtained, and we hope they can be treated as 
the current state-of-the-art performance for single-view 
ShapeNet reconstruction with a ResNet-50 image encoder.

| P | D | S | Model Description | MeanIOU (&#8593;) | Craft | Rifle | Disp. | Lamp | Speaker | Cabinet | Chair | Bench | Car | Plane | Sofa | Table | Phone | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| Pshapenet | D0disndvrdgs | S2 | (Ours) DGS Best  | 67.6 |     66.8     |   75.1   |    68.7    |   55.1  |    65.7    |    63.7    |   66.5   |   58.2   |  71.5  |     67.4    |   72.9  |   63.5   |   84.2  | 
| Pshapenet | D0disndvr | S2 | (Oracle) [DISN](https://github.com/laughtervv/DISN) [DVR](https://github.com/autonomousvision/differentiable_volumetric_rendering)   | 67.2 |   66.9     |   74.4   |    67.3    |   54.5  |    66.1    |    63.2    |   66.2   |   58.6   |  71.2  |     67.2    |   72.5  |   63.1   |   82.6  | 

Convergence requires roughly 500K iterations. 
You need a GPU with at least 20G memories.
Note that by no means 
this indicates the original setting in the previous table is inferior.
In this "our best model", we used large batch size, deeper decoder,
refined implementation of the ResNet-50 encoder. The gain mostly
comes from engineering previledges. Once again, our DGS model still
demonstrates slight preveledge when compared to the oracle baseline
even if trained with less annotations.

## Pre-trained Weights

Download the pretrained encoder from:
[DGS](https://www.dropbox.com/s/2tukeyq9lq6rcbb/encoder_net_latest.pth),
[CoReNet](https://www.dropbox.com/s/imn9jww2k759p50/encoder_net_latest.pth),
[DGS Best](https://www.dropbox.com/s/xrimy1vfpao4zeu/encoder_net_latest.pth),
[DISN DVR](https://www.dropbox.com/s/fmpt176x4o0kzla/encoder_net_latest.pth).

Download the pretrained networkTwo from:
[DGS](https://www.dropbox.com/s/6ys033ngiwbirr6/networkTwo_net_latest.pth),
[CoReNet](https://www.dropbox.com/s/lagsat78gp2y9df/networkTwo_net_latest.pth),
[DGS Best](https://www.dropbox.com/s/lmojvwcll4gy3w2/networkTwo_net_latest.pth),
[DISN DVR](https://www.dropbox.com/s/nr26gkif6gcj6tw/networkTwo_net_latest.pth).

Put the pre-trained weights according to the following structure.

```
|- $PROJ_DIR
   |- v
      |- P
         |- Ptag
            |- Dtag
               |- Stag
                  |- Rrelease
                     |- models
                        |- encoder_net_latest.pth
                        |- networkTwo_net_latest.pth
```

## Testing

Use the following command to test a model

```
cd $PROJ_DIR/src/B/Bpshapenet/
MC=PtagDtagStagRtagItag CUDA_VISIBLE_DEVICES=0 python Bpshapenet_script1_stepOneOwn12z1.py
MC=PtagDtagStagRtagItag CUDA_VISIBLE_DEVICES=0 python Bpshapenet_script2_stepBen.py
```

For the case of the pre-trained DGS model where `MC=PshapenetD0corenetdgsS2RreleaseIlatest`, the following chart is 
expected to show

```
+-------------------+---------------+----------+------------+---------+------------+------------+----------+----------+--------+-------------+---------+----------+----------+---------+
|       method      | watercraftIou | rifleIou | displayIou | lampIou | speakerIou | cabinetIou | chairIou | benchIou | carIou | airplaneIou | sofaIou | tableIou | phoneIou | meanIou |
+-------------------+---------------+----------+------------+---------+------------+------------+----------+----------+--------+-------------+---------+----------+----------+---------+
|     (Ours) DGS    |      63.3     |   70.4   |    65.7    |   49.0  |    61.2    |    57.0    |   62.1   |   50.9   |  70.1  |     62.8    |   70.3  |   56.9   |   78.3   |   62.9  |
+-------------------+---------------+----------+------------+---------+------------+------------+----------+----------+--------+-------------+---------+----------+----------+---------+
```
