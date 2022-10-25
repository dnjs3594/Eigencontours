import argparse

def parse_args(cfg):
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--mode', default='encoding', help='mode')
    parser.add_argument('--node_num', default=360, type=int help='node_num')
    parser.add_argument('--dim', default=36, type=int, help='dim')
    parser.add_argument('--display', default=False, type=bool, help='visualization')

    args = parser.parse_args()
    cfg = args_to_config(cfg, args)

    return cfg

def args_to_config(cfg, args):
    cfg.mode = args.mode
    cfg.node_num = args.node_num
    cfg.dim = args.dim
    cfg.display = args.display

    return cfg
