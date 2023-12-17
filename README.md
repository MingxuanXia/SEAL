# A Separation and Alignment Framework for Black-box Domain Adaptation

This is a PyTorch implementation of our AAAI 2024 paper SEAL.

## Start Running SEAL on Office, Office-Home, VisDA, and DomainNet.

**Step 1. Data Preparation**

Please download the [datasets](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md) and put them into ```./data``` .

**Step 2. Generate information list for dataset**

```shell
# For Office
python generate_infos.py --ds office31

# For Office-Home
python generate_infos.py --ds office_home

# For VisDA
python generate_infos.py --ds visda17

# For DomainNet
python generate_infos.py --ds DomainNet
```

**Step 3. Training black-box source domain**

```shell
# For Office on domain A
CUDA_VISIBLE_DEVICES=0 python train_src_v1.py configs/office31/src_a/train_src_a.py

# For Office-Home on domain A
CUDA_VISIBLE_DEVICES=1 python train_src_v1.py configs/office_home/src_A/train_src_A.py

# For VisDA
CUDA_VISIBLE_DEVICES=2 python train_src_v2.py configs/visda17/train_src.py

# For DomainNet
CUDA_VISIBLE_DEVICES=3 python train_src_v1.py configs/DomainNet/src_c/train_src_c.py
```

**Step 4. Adapting to target domain using SEAL**

```shell
# For Office on domain shift A->D
CUDA_VISIBLE_DEVICES=0 python train_SEAL.py configs/office31/src_a/SEAL_d.py

# For Office-Home on domain shift A->C
CUDA_VISIBLE_DEVICES=1 python train_SEAL.py configs/office_home/src_A/SEAL_C.py

# For VisDA
CUDA_VISIBLE_DEVICES=2 python train_SEAL.py configs/visda17/SEAL.py

# For DomainNet on domain shift C->P
CUDA_VISIBLE_DEVICES=3 python train_SEAL.py configs/DomainNet/src_c/SEAL_p.py
```