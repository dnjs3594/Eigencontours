![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# [CVPR 2022] Eigencontours: Novel Contour Descriptors Based on Low-Rank Approximation (Oral)
### Wonhui Park, Dongkwon Jin, and Chang-Su Kim

<img src="https://github.com/dnjs3594/Eigencontours/blob/master/overview.png" alt="overview" width="90%" height="90%" border="10"/>

Official implementation for **"Eigencontours: Novel Contour Descriptors Based on Low-Rank Approximation"** 
[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Eigencontours_Novel_Contour_Descriptors_Based_on_Low-Rank_Approximation_CVPR_2022_paper.pdf) [[arxiv]](https://arxiv.org/abs/2203.15259) [[video]](https://www.youtube.com/watch?v=nneyhWCQY-s)

### Related work
We wil also present another paper, **"Eigenlanes: Data-Driven Lane Descriptors for Structurally Diverse Lanes"**, accepted to CVPR 2022 [[paper]](https://arxiv.org/abs/2203.15302) [[arxiv]](https://arxiv.org/abs/2203.15302) [[video]](https://www.youtube.com/watch?v=XhEj3o3iihQ) [[github]](https://github.com/dongkwonjin/Eigenlanes).
Congratulations to my eigenbrother!

### Requirements
- PyTorch >= 1.4.0
- CUDA >= 10.0
- CuDNN >= 7.6.5
- python >= 3.6

### Installation
1. Download repository. We call this directory as `ROOT`:
```
$ git clone https://github.com/dnjs3594/Eigencontours.git
```

2. Download [preprocessed-data] (https://drive.google.com/file/d/12FETybRT2QdNRFuknHPM0tCzA-5ID1MB/view?usp=sharing) in `ROOT/Preprocessing/code_v1_COCO`:
```
$ cd ROOT/Preprocessing/code_v1_COCO
$ unzip data.zip
```

3. Create conda environment:
```
$ conda create -n eigencontours python=3.7 anaconda
$ conda activate eigencontours
```

4. Install dependencies:
```
$ conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
$ pip install -r requirements.txt
```

5. Optional install dependencies:
Our PolarMask is based on [mmdetection] (https://github.com/open-mmlab/mmdetection). Please check [INSTALL.md] (https://github.com/dnjs3594/Eigencontours/edit/master/mmdetection.md) for installation instructions.
```
$ pip install -r requirements_ins_seg.txt
```


### Directory structure
    .                           # ROOT
    ├── Preprocessing           # directory for data preprocessing
    │   ├── code_v1_COCO        # COCO dataset preprocessing
    │   ├── code_v1_SBD         # SBD dataset preprocessing (TBD)
    │   ├── data                # dataset storage
    │   │   ├── COCO            # COCO dataset
    │   │   │   ├──train2017
    │   │   │   ├──val2017
    │   │   │   └──annotations
    .   └── ...

### COCO dataset
Download COCO dataset in `ROOT/Preprocessing/data as directory structure:

### Preprocessing (Construct contour descriptors called "Eigencontours".)
Data preprocessing is divided into three steps, which are "encoding", "svd", "convert". Below we describe each step in detail.

1. In "encoding", star-convex contours are generated and saved to pickle format. you can set the dimensionality of the contours, N = node_num)
```
$ cd ROOT/Preprocessing/code_v1_COCO/
$ python main.py --mode encoding --node_num N (ex:360) --display False
```

2. In "svd", a contour matrix is constucted and the eigencontours are generated by SVD (Singular Value Decomposition). you can get contour descriptors, matrix "U".
```
$ cd ROOT/Preprocessing/code_v1_COCO/
$ python main.py --mode svd --display False
```

3. In "convert", coefficient vectors are generated by calculating between the star-convex contours and the eigencontours. Moreover, the F-scores of descriptors are also saved. you can set the dimensionality of coefficient vectors, M = rank-M approximation)
```
$ cd ROOT/Preprocessing/code_v1_COCO/
$ python main.py --mode encoding --dim M (ex:36) --display False
```

Optionally, you can adjust image size and threshold IoU (in "encoding") in `ROOT/Preprocessing/code_v1_COCO/options/config.py`.

In paper, objects are cropped and centerally allgned for evaluating. If you want to preprocess data in crop version, you replace "datasets.dataset_coco_not_crop" with "dataset.dataset_coco" in `ROOT/Preprocessing/code_v1_COCO/libs/prepare.py`.

### Instance segmentation (PolarMask-based method using "Eigencontours".)




### Reference
```
@Inproceedings{
    park2022eigencontours,
    title={Eigencontours: Novel Contour Descriptors Based on Low-Rank Approximation},
    author={Park, Wonhui and Jin, Dongkwon and Kim, Chang-Su},
    booktitle={CVPR},
    year={2022}
}
```
