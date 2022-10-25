# Domain Agnostic Soft Prompt 
Prompt-based learning has been a new learning paradigm which is abetter way to extract knowledge from pre-trained language model. This work is based on soft prompt tuning which can be easily updated by gradient descent to obtain the optimal prompt in a handcrafted-free way. Domain-agnostic soft prompt can be directly applied to any domains data which are unseen during training.
## Environment setup
```
pip install -r requirements.txt
```

## Dataset download
download the dataset from https://www.cs.jhu.edu/~mdredze/datasets/sentiment/ for and https://github.com/FrankWork/fudan_mtl_reviews

## Training

### domain-agnostic soft prompt
```
python train_dasp.py \
--data_path <path to dataset folder>
--dataset <amazon or mtl>
--test_domain <all domains or single domain>
--plm_type bert 
--plm bert-base-uncase
--max_seq 256
--epoch 10
--bz 8
--k_spt 8
--k_qry 8
--num_task 1000   
--task_bz 8     
--inner_lr 5e-5
--outer_lr 5e-5
--inner_step 4
```

### soft prompt
```
python train_sp.py \
```

### fine-tune
```
python train_ft.py
```
