# basic usage
python train.py --dataset cifar10 --net vgg16 --project test --name test --num_epoch 60

# adv training
python train.py --dataset cifar10 --net vgg16 --project test --name test --num_epoch 60 --train_mode adv --attack noise --sigma 0.125

python train.py --dataset cifar10 --net vgg16 --project test --name test --num_epoch 2 --train_mode adv --attack fgsm --ord inf --eps 0.3


# test neuron entropy
python train.py --dataset cifar10 --net vgg16 --project expressive --name split_0.1 --num_step 240 --val_step 120\
                --train_mode exp --split 0.1

# prune training
python train.py --dataset cifar10 --net vgg16 --project prune --name prune --num_epoch 120 --train_mode pru\
        --lr_scheduler cyclic --num_circles 4 --method Hard --batch_size 128  --prune_every 10 --activation ReLU --npbar

python train.py --dataset cifar10 --net vgg16 --project prune --name prune --num_epoch 120 --train_mode pru\
        --lr_scheduler cyclic --num_circles 4 --method RandomUnstructured --batch_size 128  --prune_every 10 --activation ReLU --npbar

python train.py --dataset cifar10 --net vgg16 --project prune --name prune --num_epoch 120 --train_mode pru\
        --lr_scheduler cyclic --num_circles 4 --method RandomUnstructured --prune_eta 0 --batch_size 128  --prune_every 10 --activation ReLU --npbar

python train.py --dataset cifar10 --net vgg16 --project prune --name exp_1 --num_epoch 120 --train_mode pru \
                --lr_scheduler cyclic --num_circles 4 --method LnStructured --prune_eta 0 --batch_size 128  --prune_every 10 --activation ReLU --npbar


# DualNet training

python train.py --dataset cifar10 --net vgg16 --project dual --name test --num_epoch 120 -- train_mode dual\
          --lr_scheduler milestone --eta_float 0.1 --eta_fixed 0 --activation ReLU