# gSDF: Geometry-Driven Signed Distance Functions for 3D Hand-Object Reconstruction (CVPR 2023)

This repository is implementation of [gSDF](https://arxiv.org/abs/2304.11970).

## Installation
Please follow instructions listed below to build the environment.
```
conda create -n gsdf python=3.9
conda activate gsdf
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```
## Dataset
1. ObMan dataset preparations. 
- Download ObMan data from [the official website](https://www.di.ens.fr/willow/research/obman/data/requestaccess.php).
- Set up a soft link from the download path to `${ROOT}/data/obman/data`.
- Download processed [SDF files](https://drive.google.com/drive/folders/1GjFJBJlbJxeYrExtcYEdhAaeH-wLZOIF) and [json files](https://drive.google.com/drive/folders/1DBzG9J0uLzCy4A6W6Uq6Aq4JNAHiiNJQ).
- Run `${ROOT}/preprocess/cocoify_obman.py` to generate LMDB training files. The data organization looks like this: 
   ```
   ${ROOT}/data/obman
   └── splits
       obman_train.json
       obman_test.json
       obman.py
       data
        ├── val
        ├── train
        |   ├── rgb
        |   ├── rgb.lmdb
        |   ├── sdf_hand
        |   ├── sdf_hand.lmdb
        |   ├── sdf_obj
        |   ├── sdf_obj.lmdb
        └── test
            ├── rgb
            ├── mesh_hand
            ├── mesh_obj
   ```

2. DexYCB dataset preparations. 
- Download DexYCB data from [the official webpage](https://dex-ycb.github.io/).
- Set up a soft link from the download path to `${ROOT}/data/dexycb/data`.
- Download processed [SDF files](https://drive.google.com/drive/folders/15yjzjYcqyOiIbX-6uaeYOezVH4stDTCG) and [json files](https://drive.google.com/drive/folders/1qULhMx1PrnXkihrPacIFzLOT5H2FZSj7).
- Run `${ROOT}/preprocess/cocoify_dexycb.py` to generate LMDB training files. The data organization looks like this: 
   ```
   ${ROOT}/data/obman
   └── splits
       toolkit
       dexycb_train_s0.json
       dexycb_test_s0.json
       dexycb.py
       data
        ├── 20200709-subject-01
        ├── .
        ├── .
        ├── 20201022-subject-10
        ├── bop
        ├── models
        ├── mesh_data
        ├── sdf_data
        ├── rgb_s0.lmdb
        ├── sdf_hand_s0.lmdb
        └── sdf_obj_s0.lmdb
   ```

## Training
1. Establish the output directory by `mkdir ${ROOT}/outputs` and `cd ${ROOT}/main`.

2. Train the gSDF model:

For Obman
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --gpu 4-7 --cfg ../asset/yaml/obman_train.yaml
```

For DexYCB
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --gpu 4-7 --cfg ../asset/yaml/dexycb_train.yaml
```

## Testing and Evaluation
Actually, when it finishes training, the script will launch the testing automatically. You could also launch the training explicitly by:\
For Obman
```
CUDA_VISIBLE_DEVICES=1 python test.py --gpu 1 --cfg ../asset/yaml/dexycb_test.yaml
```
For DexYCB
```
CUDA_VISIBLE_DEVICES=1 python test.py --gpu 1 --cfg ../asset/yaml/dexycb_test.yaml
```
After the testing phase ends, you could evaluate the performance:
For Obman
```
CUDA_VISIBLE_DEVICES=1 python eval.py --gpu 1 -e ../outputs/hsdf_osdf_2net_pa/gsdf_obman
```
For DexYCB
```
CUDA_VISIBLE_DEVICES=2 python eval.py -e ../outputs/hsdf_osdf_2net_video_pa/gsdf_dexycb_video
```