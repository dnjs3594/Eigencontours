import argparse

def parse_args(cfg):

    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--node_num', default=360, help='node_num')
    parser.add_argument('--mode', default='encoding', help='mode')
    args = parser.parse_args()

    cfg = args_to_config(cfg, args)
    return cfg

def args_to_config(cfg, args):

    cfg.node_num = args.node_num
    return cfg