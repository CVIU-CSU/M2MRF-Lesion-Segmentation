import argparse
import torch

from mmcv import Config
from mmcv.cnn import get_model_complexity_info

from mmseg.models import build_segmentor

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--print', action='store_true')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1440, 960],
        help='input image size')
    args = parser.parse_args()
    return args


def main():
    torch.backends.cudnn.benchmark = False

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    model = build_segmentor(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model = model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
                format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    memory = '{:.2f} M'.format(memory)
    split_line = '=' * 30
    if args.print:
        print(model)
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\nMemory: {4}\n{0}'.format(
        split_line, input_shape, flops, params, memory))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
