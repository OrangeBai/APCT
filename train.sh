# python imagenet_to_hdf5.py --mode clean

# python train.py  --world_size 2 --dataset imagenet --attack noise --sigma 0.15 --exp_id noise_015 --resume 0 --print_every 10 --num_epoch 15
CUDA_VISIBLE_DEVICES=2,3 python train.py  --world_size 2 --dataset imagenet --exp_id normal  --resume 0 --print_every 10 --num_epoch 15 



