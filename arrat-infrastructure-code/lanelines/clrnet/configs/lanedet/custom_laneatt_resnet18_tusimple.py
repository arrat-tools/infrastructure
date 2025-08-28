net = dict(
    type='Detector',
)
work_dirs = "work_dirs/lanedet/laneatt_r18_tusimple"

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)
featuremap_out_channel = 512 
featuremap_out_stride = 32 

num_points = 72
max_lanes = 3
sample_y=range(600, 400, -3)

heads = dict(type='LaneATT',
        anchors_freq_path='.cache/tusimple_anchors_freq.pt',
        topk_anchors=1000)

test_parameters = dict(
    conf_threshold=0.2,
    nms_thres=45,
    nms_topk=max_lanes
)

optimizer = dict(
  type = 'Adam',
  lr = 0.0003,
)

epochs = 100
batch_size = 8
total_iter = (3616 // batch_size) * epochs
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)

eval_ep = 1
save_ep = epochs

real_img_w = 2208
real_img_h = 1242
ori_img_w=1280
ori_img_h=720
img_w=640 
img_h=360
cut_height=0


val_process = [
    dict(type='GenerateLaneLine',
        transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
        ],
        training=False),  # wh=(img_w, img_h)
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
    )
)


workers = 12
log_interval = 100
seed=0
lr_update_by_epoch = False


#,
#        transforms=[
#             dict(name='Resize',
#                  parameters=dict(size=dict(height=img_h, width=img_w)),
#                  p=1.0),
#        ],