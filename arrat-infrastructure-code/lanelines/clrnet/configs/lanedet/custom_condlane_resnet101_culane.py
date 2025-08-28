net = dict(
    type='Detector',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet101',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
    in_channels=[64, 128, 256, 512]
)

sample_y = range(590, 270, -8)

batch_size = 8
aggregator = dict(
    type='TransConvEncoderModule',
    in_dim=2048,
    attn_in_dims=[2048, 256],
    attn_out_dims=[256, 256],
    strides=[1, 1],
    ratios=[4, 4],
    pos_shape=(batch_size, 10, 25),
)

neck=dict(
    type='FPN',
    in_channels=[256, 512, 1024, 256],
    out_channels=64,
    num_outs=4,
    #trans_idx=-1,
)

loss_weights=dict(
        hm_weight=1,
        kps_weight=0.4,
        row_weight=1.,
        range_weight=1.,
    )

num_lane_classes=1
heads=dict(
    type='CondLaneHead',
    heads=dict(hm=num_lane_classes),
    in_channels=(64, ),
    num_classes=num_lane_classes,
    head_channels=64,
    head_layers=1,
    disable_coords=False,
    branch_in_channels=64,
    branch_channels=64,
    branch_out_channels=64,
    reg_branch_channels=64,
    branch_num_conv=1,
    hm_idx=2,
    mask_idx=0,
    compute_locations_pre=True,
    location_configs=dict(size=(batch_size, 1, 80, 200), device='cuda:0')
)

work_dirs = "work_dirs/lanedet/condlane_r101_culane"

optimizer = dict(type='AdamW', lr=3e-4, betas=(0.9, 0.999), eps=1e-8)

epochs = 16
total_iter = (88880 // batch_size) * epochs
import math
scheduler = dict(
    type = 'MultiStepLR',
    milestones=[8, 14],
    gamma=0.1
)

seg_loss_weight = 1.0
eval_ep = 1
save_ep = 1 

img_norm = dict(
    mean=[75.3, 76.6, 77.6],
    std=[50.5, 53.8, 54.3]
)

real_img_w = 2208
real_img_h = 1242
img_height = 320 
img_width = 800
cut_height = 0 
ori_img_h = 590
ori_img_w = 1640

mask_down_scale = 4
hm_down_scale = 16
num_lane_classes = 1
line_width = 3
radius = 6
nms_thr = 4
img_scale = (800, 320)
crop_bbox = [0, 270, 1640, 590]
mask_size = (1, 80, 200)


val_process = [
    #dict(type='Alaug',
    #    transforms=[dict(type='Compose', params=dict(bbox_params=None,bboxes=False, keypoints=True, masks=False)),
    #        dict(type='Crop',
    #        x_min=crop_bbox[0],
    #        x_max=crop_bbox[2],
    #        y_min=crop_bbox[1],
    #        y_max=crop_bbox[3],
    #        p=1),
    #   dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1)]
    #),
    dict(type='CropToShape',size = crop_bbox),
    dict(type='Resize', size=(img_width, img_height)),
    #dict(type='GenerateLaneLine',
    #     transforms=[
    #         dict(name='Resize',
    #              parameters=dict(size=dict(height=img_h, width=img_w)),
    #              p=1.0),
    #     ],
    #     training=False),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = ''
dataset_type = 'custom'

dataset = dict(
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='val',
        processes=val_process,
    ),
)


workers = 12 
log_interval = 1000
lr_update_by_epoch=True
