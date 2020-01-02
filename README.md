# 3DDFA
This is a PyTorch reimplementation of the paper: [Face Alignment in Full Pose Range: A 3D Total Solution](https://arxiv.org/abs/1804.01005). 

## Dataset

MS-Celeb-1M dataset for training, 3,804,846 faces over 85,164 identities.


## Dependencies
- Python 3.6.8
- PyTorch 1.3.0

## Usage

1. Clone this repo.
2. Build cython module:
```bash
$ cd utils/cython
$ python3 setup.py build_ext -i
```

### Data preprocess
Extract images, scan them, to get bounding boxes and landmarks:
```bash
$ python3 extract.py
$ python3 pre_process.py
```

### Train
```bash
$ python3 train.py
```

To visualize the training process：
```bash
$ tensorboard --logdir=runs
```

### Demo
```bash
$ python3 demo.py
```

#### Image
![image](https://github.com/foamliu/3DDFA/raw/master/images/0008.jpg)

#### 3D Face
![image](https://github.com/foamliu/3DDFA/raw/master/images/0008.gif)