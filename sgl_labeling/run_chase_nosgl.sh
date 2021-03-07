#Training
python main.py --dataset CHASE --self_ensemble --patch_size 256 --model CON --loss 1*BCE --data_train CHASE --data_test CHASET --n_GPUs 1 --epochs 100 --data_range '1-20/1-20' --save chase_1_split1 --scale 1 --patch_size 256 --reset #--save_gt --save_results
#Inference
python main.py --dataset CHASE --test_only --patch_size 256 --model CON --data_test CHASET --n_GPUs 1 --data_range '1-20/1-20' --pre_train  '../experiment/chase_1_split1/model/model_latest.pt' --self_ensemble --scale 1 --save 'test_chase_split1' --patch_size 256 --save_gt --save_results
