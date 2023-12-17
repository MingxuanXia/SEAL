_base_ = [
    '../SEAL_base.py'
]

tau_loss = 0.3

# data
src, tgt = 'R', 'C'
info_path = f'./data/office_home_infos/{tgt}_list.txt'
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

load = f'./checkpoints/office_home/src_{src}/train_src_{src}/best_val.pth'
