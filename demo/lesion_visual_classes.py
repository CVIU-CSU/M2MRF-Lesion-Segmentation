from argparse import ArgumentParser

import PIL.Image
import imgviz
import pandas
import cv2
import os

import mmcv
import numpy as np
from mmcv.runner import load_checkpoint

from mmseg.apis import inference_segmentor
from mmseg.models import build_segmentor

LINE_WIDTH = 4
COLORS = [
    (0, 0, 255),  # EX: red
    (0, 255, 0),  # HE: green
    (0, 255, 255),  # SE: yellow
    (255, 0, 0)  # MA: blue
]
classes = ['EX', 'HE', 'SE', 'MA']


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.test_cfg.compute_aupr = False  # NEW
    model = build_segmentor(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def draw_label(result):
    n_classes, h, w = result.shape
    res = np.zeros((h, w), dtype=np.int32)
    for i in range(n_classes):
        index = result[i] > 0
        res[index] = i + 1

    res = PIL.Image.fromarray(res.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    res.putpalette(colormap)
    return res


def draw_edges(img_path, result):
    image = cv2.imread(img_path)
    n_classes, h, w = result.shape

    for i in range(n_classes):
        label = result[i]
        label = label.astype(np.uint8)
        contours, _ = cv2.findContours(label, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.drawContours(image, contours, -1, COLORS[i], thickness=LINE_WIDTH)

    return image


def compute_iou(pred, gt):
    pred = pred.astype('bool')
    gt = gt.astype('bool')

    tp = np.sum(gt & pred)
    fn = np.sum(gt & ~pred)
    fp = np.sum(~gt & pred)

    iou = tp / (tp + fp + fn)
    iou = np.nan_to_num(iou, nan=0)
    return iou


def generate_visual_mask(pred, gt):
    h, w = gt.shape
    pred = pred.astype('int') * 255
    gt = gt.astype('int') * 255
    pred = np.expand_dims(pred, axis=2)
    gt = np.expand_dims(gt, axis=2)
    visual_mask = np.concatenate((np.zeros((h, w, 1)), gt, pred), axis=2)

    return visual_mask


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('-d', help='dataset', default='IDRID')
    parser.add_argument('-o', help='output dir', default='baseline')
    parser.add_argument('-s', help='short name', default='')
    args = parser.parse_args()

    image_root = f'../data/{args.d}/image/test/'
    label_root = f'../data/{args.d}/label/test/annotations'

    output_root = os.path.join('../visual/', args.o)
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, 'visual'), exist_ok=True)
    for c in classes:
        os.makedirs(os.path.join(output_root, c), exist_ok=True)

    output_xlsx = dict()
    output_xlsx['filename'] = list()
    for c in classes:
        output_xlsx[c] = list()

    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    for root, dirs, files in os.walk(image_root):
        for file in files:
            print(file)
            img_path = os.path.join(root, file)
            result = inference_segmentor(model, img_path)  # result: (c,h,w)

            filename = file.split('.')[0]
            label_path = os.path.join(label_root, filename + '.png')
            label = PIL.Image.open(label_path)
            label = np.array(label)

            output_xlsx['filename'].append(filename)
            for i in range(len(classes)):
                name = classes[i]
                pred = result[i]
                gt = label == (i + 1)
                visual_mask = generate_visual_mask(pred, gt)
                cv2.imwrite(os.path.join(output_root, name, f'{filename}_{name}_{args.s}.jpg'), visual_mask)

                iou = compute_iou(pred, gt)
                output_xlsx[name].append(100 * iou)

            label = draw_label(result)
            label.save(os.path.join(output_root, '{}_{}_label.png'.format(filename, args.o)))

            edge = draw_edges(img_path, result)
            cv2.imwrite(os.path.join(output_root, 'visual', '{}_{}_edge.jpg'.format(filename, args.s)), edge)

    df = pandas.DataFrame(output_xlsx)
    df.to_excel(os.path.join(output_root, f'iou_{args.s}.xlsx'), index=False, encoding='utf-8')

    print('Saved to {}'.format(output_root))


if __name__ == '__main__':
    main()
