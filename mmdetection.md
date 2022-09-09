## Installation

### Requirements

- Linux
- Python 3.5+
- PyTorch 1.0+ or PyTorch-nightly
- CUDA 9.0+
- NCCL 2+
- GCC 4.9+

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.0/9.2/10.0
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC: 4.9/5.3/5.4/7.3

### Install mmdetection

a. Move instance segmentation root and install Cython.

```shell
cd Root/Instance_segmentation/code_v1_COCO
conda install cython
pip install mmcv==0.2.14
```

b. Clone the mmdetection repository.

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```

c. Install mmdetection (other dependencies will be installed automatically).

```shell
python setup.py develop
# or "pip install -v -e ."
```

d. Install other modules.

```shell
cd ..
python setup.py develop
# or "pip install -v -e ."
```

### Notice
You can run `python(3) setup.py develop` or `pip install -v -e .` to install mmdetection if you want to make modifications to it frequently.

If there are more than one mmdetection on your machine, and you want to use them alternatively.
Please insert the following code to the main file
```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```
or run the following command in the terminal of corresponding folder.
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```
