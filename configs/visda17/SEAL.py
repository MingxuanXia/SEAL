# separation module
tau_loss = 0.4
tau_conf = 0.95
tau_scale = 0.5
upd_tau_trans = 1

# alignment module
temperature = 0.07
knn = 100
knn_start = 3
knn_weight = 1.0

# loss module
ur_weight = 1.0
cont_weight = 0.5
div = True
is_cr = True
is_mu = True

# ISLAND hyper-parameters
tau_loss = 0.4
tau_conf = 0.95
tau_scale = 0.5
upd_tau_trans = 1

# model
num_classes = 12
src_model = dict(type='ResNet', depth=101, num_classes=num_classes)
tgt_model = dict(type='ResNet', depth=101, num_classes=num_classes, pretrained=True)
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
src, tgt = 't', 'v'
info_path = f'./data/visda17_infos/{tgt}_list.txt'
test_class_acc = True

batch_size = 64
num_workers = 4
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
data = dict(
    train=dict(
        ds_dict=dict(
            type='SubVisda17',
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
            type='SubVisda17',
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
            type='SubVisda17',
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
            type='SubVisda17',
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
epochs = 20
warmup_epochs = 2
CE_warmup = True
lr = 0.01
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

# log & save
log_interval = 100
test_interval = 1
pred_interval = 1
work_dir = None
resume = None
load = f'./checkpoints/visda17/train_src/best_val.pth'
port = 10001
