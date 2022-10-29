# python imagenet_to_hdf5.py --mode clean

<<<<<<< HEAD
<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=0 python train.py  --world_size 1 --dataset imagenet --attack noise --sigma 0.10 --exp_id noise_010 --resume 1 --num_epoch 30 --net resnet50 --print_every 10 --phase_path imagenet_cfg/phase_res.yml 
CUDA_VISIBLE_DEVICES=1 python train.py  --world_size 1 --dataset imagenet --attack noise --sigma 0.0 --exp_id std --resume 1 --num_epoch 30 --net resnet50 --print_every 10 --phase_path imagenet_cfg/phase_res.yml 
# CUDA_VISIBLE_DEVICES=2 python train.py  --world_size 1 --dataset imagenet --attack noise --sigma 0.10 --exp_id noise_010 --resume 0  --num_epoch 15 --net vgg13 --print_every 50
# CUDA_VISIBLE_DEVICES=3 python train.py  --world_size 1 --dataset imagenet --attack noise --sigma 0.0 --exp_id std --resume 0 --num_epoch 15 --net vgg13 --print_every 5


CUDA_VISIBLE_DEVICES=2,3 python train.py  --world_size 2 --dataset imagenet --attack noise --sigma 0.0 --exp_id std --resume 1 --num_epoch 33 --net resnet50 --print_every 100 --phase_path imagenet_cfg/phase_res.yml
=======
python train.py  --world_size 2 --yml_file ./imagenet_cfg/p_0.yml --attack noise --sigma 0.15 --exp_id noise_015 --resume 0
python train.py  --world_size 2 --yml_file ./imagenet_cfg/p_1.yml --attack noise --sigma 0.15 --exp_id noise_015 --resume 1
=======
python train.py  --world_size 2 --dataset imagenet --attack noise --sigma 0.15 --exp_id noise_015 --resume 0 --print_every 10 --num_epoch 15
# CUDA_VISIBLE_DEVICES=2,3 python train.py  --world_size 2 --dataset imagenet --exp_id normal  --resume 0 --print_every 10 --num_epoch 15 

>>>>>>> 3078b1d (.)


>>>>>>> 8e9875c (.)
