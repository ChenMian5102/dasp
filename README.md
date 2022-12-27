# Domain Agnostic Soft Prompt 
Prompt-based learning has been a new learning paradigm which is a better way to extract knowledge from pre-trained language model. This work is based on soft prompt tuning which can be easily updated by gradient descent to obtain the optimal prompt in a handcrafted-free way. Domain-agnostic soft prompt can be directly applied to any domains data which are unseen during training.
## Environment setup
```
pip install -r requirements.txt
```

## Dataset download
download the dataset from https://github.com/FrankWork/fudan_mtl_reviews

## Training

### domain-agnostic soft prompt
```
python train_dasp.py \
    --data_path mtl-dataset/ \
    --dataset amazon \
    --test_domain all \
    --plm_type bert \
    --plm bert-base-uncased \
    --tune_plm False \
    --max_seq 256 \
    --epoch 10 \
    --bz 8 \
    --k_spt 8 \
    --k_qry 8 \
    --num_task 1000 \
    --task_bz 8 \
    --inner_lr 5e-5 \
    --outer_lr 5e-5 \
    --inner_step 4
```

### meta soft prompt
```
python train_metasp.py \
    --data_path mtl-dataset/ \
    --test_domain all \
    --plm_type bert \
    --plm bert-base-uncased \
    --max_seq 256 \
    --epoch 10 \
    --bz 8 \
    --k_spt_label 8 \
    --k_spt_unlabel 8 \
    --k_qry 8 \
    --num_task 1000 \
    --task_bz 8 \
    --inner_lr 5e-3 \
    --outer_lr 1e-3 \
    --inner_step 4 \
    --mlm_lambda 0.5 
```


<!-- ### soft prompt
```
python train_sp.py \
```

### fine-tune
```
python train_ft.py
``` -->
