_base_ = [
    '../SEAL_base.py'
]

tau_loss = 0.3

# data
src, tgt = 's', 'r'
info_path = f'./data/DomainNet_infos/{tgt}_list.txt'
data = dict(
    warmup=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    eval_train=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    test=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    train=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
)

load = f'./checkpoints/DomainNet/src_{src}/train_src_{src}/best_val.pth'
