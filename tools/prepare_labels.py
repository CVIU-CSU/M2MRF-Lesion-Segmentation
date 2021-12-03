import os

import cv2
import numpy as np
import PIL.Image
import imgviz

"""
input dataset structure (e.g. IDRID):

IDRID
|-- image
|   |-- test
|   `-- train
`-- label
    |-- test
    |   |-- EX
    |   |-- HE
    |   |-- SE
    |   `-- MA
    `-- train
        |-- EX
        |-- HE
        |-- SE
        `-- MA
"""

root = '../data'
datasets = ['IDRID', 'DDR']
sets = ['train', 'val', 'test']
classes = ['EX', 'HE', 'SE', 'MA']

if len(classes) == 4:
    output_dir = 'annotations'
else:
    output_dir = 'annotations_' + '_'.join(classes)


def generate_annotations(file, image_path, label_root):
    image = cv2.imread(os.path.join(image_path, file + '.jpg'), flags=0)
    shape = image.shape
    ann = np.zeros(shape, dtype=np.int32)

    for i, c in enumerate(classes):
        label_path = os.path.join(label_root, c, file + '.png')
        if not os.path.exists(label_path):
            continue

        label = cv2.imread(label_path, flags=0)
        ann[label > 0] = i + 1

    ann_pil = PIL.Image.fromarray(ann.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()

    ann_pil.putpalette(colormap)
    return ann_pil


if __name__ == '__main__':
    for dataset in datasets:
        for s in sets:
            image_path = '{}/{}/image/{}/'.format(root, dataset, s)
            label_path = '{}/{}/label/{}/'.format(root, dataset, s)
            output_path = '{}/{}/label/{}/{}/'.format(root, dataset, s, output_dir)
            if not os.path.exists(image_path):
                continue

            os.makedirs(output_path, exist_ok=True)

            for file in os.listdir(image_path):
                print(file)
                file = file.split('.')[0]
                ann_image = generate_annotations(file, image_path, label_path)
                ann_image.save(os.path.join(output_path, file + '.png'))
