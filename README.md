![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# [CVPR 2022] Eigencontours: Novel Contour Descriptors Based on Low-Rank Approximation (Oral)
### Wonhui Park, Dongkwon Jin, and Chang-Su Kim

<img src="https://github.com/dnjs3594/Eigencontours/blob/master/overview.png" alt="overview" width="90%" height="90%" border="10"/>

Official implementation for **"Eigencontours: Novel Contour Descriptors Based on Low-Rank Approximation"** 
[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Eigencontours_Novel_Contour_Descriptors_Based_on_Low-Rank_Approximation_CVPR_2022_paper.pdf) [[arxiv]](https://arxiv.org/abs/2203.15259) [[video]](https://www.youtube.com/watch?v=nneyhWCQY-s)

### Related work
We wil also present another paper, **"Eigenlanes: Data-Driven Lane Descriptors for Structurally Diverse Lanes"**, accepted to CVPR 2022 (oral) [[paper]](https://arxiv.org/abs/2203.15302) [[supp]](https://drive.google.com/file/d/1nRqSsf2bBDAA_s5XZ_BuKPyuHEr3OHJt/view?usp=sharing) [[video]](https://www.youtube.com/watch?v=XhEj3o3iihQ).
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

### Preprocessing (Construct contour descriptors called "Eigencontours".)

Data preprocessing is divided into three steps, which are "encoding", "svd", "convert". Below we describe each step in detail.
1. In "encoding", the star-convex contours are generated and saved to pickle format. you can set the dimensionality of the contours, N = node_num)
```
$ cd ROOT/Preprocessing/code_v1_COCO/
$ python main.py --dataset_dir /where/is/your/dataset/path/
```

2. In P01, each lane in a training set is represented by 2D points sampled uniformly in the vertical direction.
3. In P02, lane matrix is constructed and SVD is performed. Then, each lane is transformed to its coefficient vector.
4. In P03, clustering is performed to obtain lane candidates.
5. In P04, training labels are generated to train the SI module in the proposed SIIC-Net.

If you want to get the preproessed data, please run the preprocessing codes in order. Also, you can download the preprocessed data.


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
