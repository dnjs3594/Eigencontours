from __future__ import division
import argparse

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from tools.args import parse_args_train

from IPython import embed

import warnings
warnings.filterwarnings('ignore')

def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

def main():
    args = parse_args_train()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)

    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.data_root is not None:
        cfg.data_root = args.data_root
        cfg.data.train.ann_file = cfg.data_root + 'annotations/instances_train2017.json'
        cfg.data.train.img_prefix = cfg.data_root + 'train2017/'
    cfg.gpus = 1

    if args.out_dir is not None:
        mkdir(args.out_dir)
        cfg.out_dir = args.out_dir

    distributed = False

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        logger=logger)


if __name__ == '__main__':
    # args = parse_args()
    # if not args.debug:
    #     main()
    # else:
    #     test()
    main()