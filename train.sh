# python imagenet_to_hdf5.py --mode clean

# CUDA_VISIBLE_DEVICES=0 python train.py  --world_size 1 --dataset imagenet --attack noise --sigma 0.10 --exp_id noise_010 --resume 0 --num_epoch 15 --net resnet50  --phase_path './imagenet_cfg/phase_res.yml' --print_every 50
# CUDA_VISIBLE_DEVICES=1 python train.py  --world_size 1 --dataset imagenet --attack noise --sigma 0.0 --exp_id std --resume 0 --num_epoch 15 --net resnet50  --phase_path './imagenet_cfg/phase_res.yml' --print_every 50
# CUDA_VISIBLE_DEVICES=2 python train.py  --world_size 1 --dataset imagenet --attack noise --sigma 0.10 --exp_id noise_010 --resume 0  --num_epoch 15 --net vgg13 --print_every 50
CUDA_VISIBLE_DEVICES=3 python train.py  --world_size 1 --dataset imagenet --attack noise --sigma 0.0 --exp_id std --resume 0 --num_epoch 15 --net vgg13 --print_every 5


