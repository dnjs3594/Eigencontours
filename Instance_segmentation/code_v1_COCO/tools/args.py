import argparse
import os

def parse_args_train():
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--work_dir', default='../work_dirs_eigen_768_1x_r50/', help='work_dir')
    parser.add_argument('--config', default='../configs/eigencontours/eigen_768_1x_r50.py', help='cfg_dir')
    parser.add_argument('--data_root', default='', help='data_root')
    parser.add_argument('--resume_from', default=False, help='resume')

    args = parser.parse_args()
    args = args_fixed(args, 'train')

    return args

def parse_args_test():
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--work_dir', default='../work_dirs_eigen_768_1x_r50/', help='work_dir')
    parser.add_argument('--checkpoint', default='../checkpoints/eigen_768_1x_r50.pth', help='ckpt_dir')
    parser.add_argument('--config', default='../configs/eigencontours/eigen_768_1x_r50.py', help='cfg_dir')
    parser.add_argument('--data_root', default='', help='data_root')
    parser.add_argument('--show', default=False, help='show')

    args = parser.parse_args()
    args = args_fixed(args, 'test')

    return args

def args_fixed(args, mode='train'):
    args.gpus = 1
    if mode == "train":
        args.out_dir = os.path.join(args.work_dir, "train_vis")
    else:
        args.out_dir = os.path.join(args.work_dir, "test_vis")
        args.out = "test.pkl"
        args.json_out = "test.json"
        args.eval = ["segm"]

    return args
