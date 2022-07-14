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
- PyTorch >= 1.6
- CUDA >= 10.0
- CuDNN >= 7.6.5
- python >= 3.6

### Installation
1. Download repository. We call this directory as `ROOT`:
```
$ git clone https://github.com/dnjs3594/Eigencontours.git
```

2. Download [preprocessed-data] (https://drive.google.com/file/d/12FETybRT2QdNRFuknHPM0tCzA-5ID1MB/view?usp=sharing) in `ROOT`:
```
$ cd ROOT
$ unzip pretrained.zip
$ unzip preprocessed.zip
```
4. Create conda environment:
```
$ conda create -n eigenlanes python=3.7 anaconda
$ conda activate eigenlanes
```
4. Install dependencies:
```
$ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
$ pip install -r requirements.txt
```



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
