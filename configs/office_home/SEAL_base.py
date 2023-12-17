# separation module
tau_loss = 0.4
tau_conf = 0.95
tau_scale = 0.1
upd_tau_trans = 1

# alignment module
temperature = 0.07
knn = 3
knn_start = 20
knn_weight = 0.2

# loss module
ur_weight = 0.2
cont_weight = 0.5
div = True
is_cr = True
is_mu = True

# model
num_classes = 65
src_model = dict(type='ResNet', depth=50, num_classes=num_classes)
tgt_model = dict(type='ResNet', depth=50, num_classes=num_classes, pretrained=True)
low_dim = 128
moco_queue = 1024
moco_m = 0.999
proto_m = 0.99
tgt_head = dict(type='BottleNeckMLP_Cont', feature_dim=2048, bottleneck_dim=256, num_classes=num_classes,
                low_dim=low_dim, type1='bn', type2='wn')

# DINE hyper-parameters
ema = 0.6
topk = 1

loss = dict(
    train=dict(type='SmoothCE'),
    test=dict(type='CrossEntropyLoss'),
)

# data
src, tgt = 'A', 'C'  # A: Art, C: Clipart, P: Product, R: Real_World
info_path = f'./data/office_home_infos/{tgt}_list.txt'
batch_size = 64
num_workers = 4
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
data = dict(
    train=dict(
        ds_dict=dict(
            type='SubOfficeHome',
            info_path=info_path,
            mode='train'
        ),
        trans_dict=dict(
            type='OHMultiView', views='qk',
            mean=mean, std=std
        )
    ),
    warmup=dict(
        ds_dict=dict(
            type='SubOfficeHome',
            info_path=info_path,
            mode='warmup'
        ),
        trans_dict=dict(
            type='OHMultiView', views='w',
            mean=mean, std=std,
        )
    ),
    eval_train=dict(
        ds_dict=dict(
            type='SubOfficeHome',
            info_path=info_path,
            mode='eval_train'
        ),
        trans_dict=dict(
            type='OHMultiView', views='t',
            mean=mean, std=std
        )
    ),
    test=dict(
        ds_dict=dict(
            type='SubOfficeHome',
            info_path=info_path,
            mode='test'
        ),
        trans_dict=dict(
            type='OHMultiView', views='t',
            mean=mean, std=std
        )
    ),
)

# training optimizer & scheduler
epochs = 100
warmup_epochs = 10
CE_warmup = False
lr = 0.01
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

# log & save
log_interval = 25
test_interval = 1
pred_interval = epochs // 10
work_dir = None
resume = None
load = f'./checkpoints/office_home/src_{src}/train_src_{src}/best_val.pth'
port = 10001