# basic usage
python train.py --dataset cifar10 --net vgg16 --project test --name test --num_epoch 60

# adv training
python train.py --dataset cifar10 --net vgg16 --project test --name test --num_epoch 60\
                --train_mode adv --attack noise --sigma 0.125

python train.py --dataset cifar10 --net vgg16 --project test --name test --num_epoch 60\
                --train_mode adv --attack fgsm --ord inf --eps 0.3


# test neuron entropy
python train.py --dataset cifar10 --net vgg16 --project expressive --name split_0.1 --num_step 24000 --val_step 400\
                --train_mode exp --split 0.1