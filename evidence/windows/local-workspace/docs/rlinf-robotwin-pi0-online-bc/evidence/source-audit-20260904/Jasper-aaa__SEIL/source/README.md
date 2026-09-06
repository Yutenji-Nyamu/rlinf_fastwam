# SEIL
The implementation of SEIL

Our implementation is mainly based on BAKU(https://github.com/siddhanthaldar/BAKU). 

## Installation

```
cd SEIL
conda env create -f conda_env.yml
```

## Data
Use the instructions in BAKU to prepare your LIBERO data
https://github.com/siddhanthaldar/BAKU/blob/main/Instructions.md

## Evolving 

Follow the _Train fewshot model -> Train selector -> Record -> Select -> Train_ loop.

To train the k-shot model in libero_long:

```
cd baku
python train.py num_demos_per_task=k use_ema=True agent=baku suite=libero dataloader=libero suite/task=libero_10 suite.hidden_dim=256 
```

And train the k-shot selector, modify the config in select/config.yaml set the data_path to your data path.
You can set the expert_shot to k for training the k-shot selector.
```
cd select
python train.py 
```

And record using environment-level augmentation after traning.

```
python record.py env_aug=True bc_weight=/path/to/weight record_path=/path/to/record agent=baku suite=libero dataloader=libero suite/task=libero_10 suite.hidden_dim=256 
```

Using select/inference.py to select next roung traning demonstrations.
Modify the save_dir and model_weight, num_to_save and select_mode in the select/config_inf.yaml 
```
python inference.py
```

And train the second round model based on same process
set the saved demonstrations path in train config to save_dir and set the use_save_demo to True 

## EVAL 

use the eval command and benchmarking to evaluate the model in LIBERO benchmark
```
python eval.py benchmarking=True agent=baku suite=libero dataloader=libero suite/task=libero_10 suite.hidden_dim=256 bc_weight=/path/to/weight
```

You can find the main changes in the following files

```
Added comprehensive libero data processing in config and dataset;

Added Record python script; # You can record your own initial states

We modified the dataset script in baku/read_dara/* to accommodate the generated demonstrations.

We Implement the Light-weight Selector in 'select' folder
```



