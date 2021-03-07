#Training
python main.py --self_ensemble --patch_size 256 --model CON --loss 1*BCE --data_train DRIVE --data_test DRIVET --n_GPUs 1 --epochs 100 --data_range '1-20/1-20' --save drive_1_split1 --scale 1 --patch_size 256 --reset #--save_gt --save_results
#Inference
python main.py --test_only --patch_size 256 --model CON --data_test DRIVET --n_GPUs 1 --data_range '1-20/1-20' --pre_train  '../experiment/drive_1_split1/model/model_latest.pt' --self_ensemble --scale 1 --save 'test_drive_split1' --patch_size 256 --save_gt --save_results
