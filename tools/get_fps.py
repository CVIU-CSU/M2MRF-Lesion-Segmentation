import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--num', type=int, default=200)
    parser.add_argument('--log-interval', type=int, default=50, help='interval of logging')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    print(args.config)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    total_iters = args.num

    # benchmark with 200 image and take the average
    end_flag = False
    total_i = 0
    while True:
        for i, data in enumerate(data_loader):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                model(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed
                total_i += 1
                if total_i % args.log_interval == 0:
                    fps = 1. * total_i / pure_inf_time
                    print(f'Done image [{total_i:<3}/ {total_iters}], '
                          f'fps: {fps:.2f} img / s')

            if total_i == total_iters:
                fps = 1. * total_i / pure_inf_time
                print(f'Overall fps: {fps:.2f} img / s')
                end_flag = True
                break

        if end_flag:
            break


if __name__ == '__main__':
    main()
