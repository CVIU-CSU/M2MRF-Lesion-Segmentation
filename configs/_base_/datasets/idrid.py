# dataset settings
"""
rgb mean:
 [116.51282647  56.43716432  16.30857136]
rgb std:
 [80.20605713 41.23209693 13.29250962]
"""
dataset_type = 'LesionDataset'
data_root = '../data/IDRID'
img_norm_cfg = dict(
    mean=[116.513, 56.437, 16.309], std=[80.206, 41.232, 13.293], to_rgb=True)
image_scale = (1440, 960)
crop_size = (960, 1440)
palette = [
    [0, 0, 0],
    [128, 0, 0],  # EX: red
    [0, 128, 0],  # HE: green
    [128, 128, 0],  # SE: yellow
    [0, 0, 128]  # MA: blue
]
classes = ['bg', 'EX', 'HE', 'SE', 'MA']
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=image_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_dir='image/train',
        ann_dir='label/train/annotations',
        data_root=data_root,
        classes=classes,
        palette=palette,
        type=dataset_type,
        pipeline=train_pipeline),
    val=dict(
        img_dir='image/test',
        ann_dir='label/test/annotations',
        data_root=data_root,
        classes=classes,
        palette=palette,
        type=dataset_type,
        pipeline=test_pipeline),
    test=dict(
        img_dir='image/test',
        ann_dir='label/test/annotations',
        data_root=data_root,
        classes=classes,
        palette=palette,
        type=dataset_type,
        pipeline=test_pipeline))
