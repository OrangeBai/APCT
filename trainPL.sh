# python imagenet_to_hdf5.py --mode clean
#
#python train.py  --world_size 2 --yml_file ./imagenet_cfg/p_0.yml --attack noise --sigma 0.15 --exp_id noise_015 --resume 0
#
#python train.py  --world_size 2 --yml_file ./imagenet_cfg/p_1.yml --attack noise --sigma 0.15 --exp_id noise_015 --resume 1

for conv_pru in 0.01 0.05 0.1 0.15
do
	for fc_pru in 0.01 0.05 0.1 0.15
	do
		for prune_every in 10 15 20
		do
		echo hard_${conv_pru}_${fc_pru}_${prune_every}
		done
	done
done