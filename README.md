# CUMnoUIL


### How to run

1. Install the required package according to `requirements.txt`

2. Create a folder `../data` and put the data naming with the parameter "dataset" in it

```shell

python main.py --dataset crosssite --rand_split_class --label_num_per_class 20 --metric acc --method CUMnoUIL --lr 0.001 --weight_decay 0. --num_layers 3 --hidden_channels 64 --num_heads 1 --rb_order 1 --rb_trans identity --lamda 0 --M 50 --K 5 --use_bn --use_residual --use_gumbel --use_act --use_jk --batch_size 10000 --runs 5 --epochs 1000 --eval_step 9 --device 0 --dropout 0.3 --filter1 1 --filter2 1 --knn_num 30 --save_init_edge --init_edge_method 'ks' --save_model --scorer_type 'MLP' --save_index 'standard' --score_thold 0.9

```