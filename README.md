# A Separation and Alignment Framework for Black-box Domain Adaptation

## An example of running SEAL on Office, Office-Home, and VisDA.

**Step 1. Data Preparation**

Please download the dataset from the official website (https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code)(http://ai.bu.edu/visda-2017/) and put them into ```./data``` .

**Step 2. Generate information list for dataset**

```shell
# For Office
python generate_infos.py --ds office31

# For Office-Home
python generate_infos.py --ds office_home

# For VisDA
python generate_infos.py --ds visda17
```

**Step 3. Training black-box source domain**

```shell
# For Office on domain A
CUDA_VISIBLE_DEVICES=0 python train_src_v1.py configs/office31/src_a/train_src_a.py

# For Office-Home on domain A
CUDA_VISIBLE_DEVICES=1 python train_src_v1.py configs/office_home/src_A/train_src_A.py

# For VisDA
CUDA_VISIBLE_DEVICES=2 python train_src_v2.py configs/visda17/train_src.py
```

**Step 4. Adapting to target domain using SEAL**

```shell
# For Office on domain shift A->D
CUDA_VISIBLE_DEVICES=0 python train_SEAL.py configs/office31/src_a/SEAL_d.py

# For Office-Home on domain shift A->C
CUDA_VISIBLE_DEVICES=1 python train_SEAL.py configs/office_home/src_A/SEAL_C.py

# For VisDA
CUDA_VISIBLE_DEVICES=2 python train_SEAL.py configs/visda17/SEAL.py
```