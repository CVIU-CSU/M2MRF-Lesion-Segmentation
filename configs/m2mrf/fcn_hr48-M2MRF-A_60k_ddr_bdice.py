_base_ = [
    '../_base_/models/fcn_hr48.py',
    '../_base_/datasets/ddr.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_60k_ddr.py'
]
model = dict(
    use_sigmoid=True,
    backbone=dict(
        type='HRNet_M2MRF_A',  # DownSample/UpSample: One-Step/One-Step
        m2mrf_patch_size=(8, 8),
        m2mrf_encode_channels_rate=4,
        m2mrf_fc_channels_rate=64,
    ),
    decode_head=dict(
        num_classes=4,
        loss_decode=dict(type='BinaryLoss', loss_type='dice', loss_weight=1.0, smooth=1e-5)
    )
)
test_cfg = dict(mode='whole', compute_aupr=True)
