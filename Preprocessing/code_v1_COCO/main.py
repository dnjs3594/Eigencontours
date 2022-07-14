import torch.nn.parallel
import torch.optim

from options.config import Config
from options.args import *
from libs.prepare import *
from libs.S1_encoding import *
from libs.S2_svd import *
from libs.S3_convert import *


def run_encoding(cfg, dict_DB):
    contour_generator = Generate_Contour(cfg, dict_DB)
    contour_generator.run()

def run_svd(cfg, dict_DB):
    svd = SVD(cfg, dict_DB)
    svd.run()

def run_convert(cfg, dict_DB):
    convertor = Convert(cfg, dict_DB)
    convertor.run()

def main():
    # option
    cfg = Config()
    cfg = parse_args(cfg)

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    torch.backends.cudnn.deterministic = True

    # prepare
    dict_DB = dict()
    dict_DB = prepare_visualization(cfg, dict_DB)
    dict_DB = prepare_dataloader(cfg, dict_DB)

    # run
    if cfg.mode == "encoding":
        run_encoding(cfg, dict_DB)
    elif cfg.mode == "svd":
        run_svd(cfg, dict_DB)
    elif cfg.mode == "convert":
        run_convert(cfg, dict_DB)
    else:
        print("Please mode check!")

if __name__ == '__main__':
    main()