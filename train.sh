# python imagenet_to_hdf5.py --mode clean

python train.py  --world_size 2 --yml_file ./imagenet_cfg/p_0.yml --attack noise --sigma 0.15 --exp_id noise_015 --resume 0

python train.py  --world_size 2 --yml_file ./imagenet_cfg/p_1.yml --attack noise --sigma 0.15 --exp_id noise_015 --resume 1
