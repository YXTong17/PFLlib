#!/bin/bash
cd ./system || exit


#################### FedAvg ####################
# nohup python -u main.py -algo FedAvg -data TinyImagenet -ncl 200 \
#   -lr 0.1 -lbs 50 -ls 5 -gr 2000 -m ResNet18 -did 0 \
#   > ../outputs/lr-0.1_lbs-50_ls-5/FedAvg/TinyImagenet_FedAvg_ResNet18-BN.out 2>&1 &



#################### FedFGAC ####################
# nohup python -u main.py -algo FedFGAC -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC/Cifar10_FedFGAC_LSR0.001_Log_CNN.out 2>&1 &

# nohup python -u main.py -algo FedFGAC -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC/Cifar100_FedFGAC_LSR0.1_Log_CNN.out 2>&1 &



#################### FedFGAC_CC ####################
# nohup python -u main.py -algo FedFGAC_CC -data TinyImagenet -ncl 200 \
#   -lr 0.1 -lbs 50 -ls 5 -gr 2000 -m ResNet18 -did 0 -al 1 \
#   > ../outputs/lr-0.1_lbs-50_ls-5/FedFGAC_CC/TinyImagenet_FedFGAC_CC_-logL1_al1_Log_ResNet18-BN.out 2>&1 &



#################### FedCC ####################
# nohup python -u main.py -algo FedCC -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -al 1 \
#   > ../outputs/lr-0.005_lbs-10/FedCC/Cifar10_FedCC-L1-1000_al1_Log_CNN.out 2>&1 &

# nohup python -u main.py -algo FedCC -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -al 1 \
#   > ../outputs/lr-0.005_lbs-10/FedCC/Cifar100_FedCC_L1-100_al1_Log_CNN.out 2>&1 &

# nohup python -u main.py -algo FedCC -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -al 100 \
#   > ../outputs/lr-0.005_lbs-10/FedCC/Cifar10_FedCC-L2_al100_Log_CNN.out 2>&1 &

# nohup python -u main.py -algo FedCC -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -al 100 \
#   > ../outputs/lr-0.005_lbs-10/FedCC/Cifar100_FedCC_L2_al100_Log_CNN.out 2>&1 &

# nohup python -u main.py -algo FedCC -data MNIST-dir100 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -al 100 \
#   > ../outputs/lr-0.005_lbs-10/FedCC/MNIST_test.out 2>&1 &

# nohup python -u main.py -algo FedCC -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -al 10 \
#   > ../outputs/lr-0.005_lbs-10/FedCC/Cifar100_FedCC-Dot_al100_Log_CNN.out 2>&1 &


# nohup python -u main.py -algo FedCC -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m ResNet18 -did 0 -al 1 \
#   > ../outputs/lr-0.005_lbs-10/FedCC/Cifar100_FedCC_L1-100_al1_Log_ResNet18-GN.out 2>&1 &










#################### FedDyn ####################
# nohup python -u main.py -algo FedDyn -data TinyImagenet -ncl 200 \
#   -lr 0.1 -lbs 50 -ls 5 -gr 2000 -m ResNet18 -did 0 -al 0.01 \
#   > ../outputs/lr-0.1_lbs-50_ls-5/FedDyn/TinyImagenet_FedDyn_al-0.01_ResNet18-BN.out 2>&1 &



#################### FedProto ####################
# nohup python -u main.py -algo FedProto -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -lam 0.1 \
#   > ../outputs/lr-0.005_lbs-10/FedProto/Cifar10_FedProto_lam0.1_CNN.out 2>&1 &

# nohup python -u main.py -algo FedProto_CR -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -lam 10 \
#   > ../outputs/lr-0.005_lbs-10/FedProto/Cifar10_FedProto_CR_lam10_CNN.out 2>&1 &

# nohup python -u main.py -algo FedProto -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -lam 1 \
#   > ../outputs/lr-0.005_lbs-10/FedProto/Cifar100_FedProto_CNN.out 2>&1 &

# nohup python -u main.py -algo FedProto_CR -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -lam 1 \
#   > ../outputs/lr-0.005_lbs-10/FedProto/Cifar100_FedProto_CR_lam1_CNN.out 2>&1 &



#################### FedTGP ####################
# nohup python -u main.py -algo FedTGP -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -se 100 -lam 0.1 -mart 100 \
#   > ../outputs/lr-0.005_lbs-10/FedTGP/Cifar10_FedTGP_lam0.1.out 2>&1 &



#################### FedALA ####################
# nohup python -u main.py -algo FedALA -data Cifar10 -ncl 10 \
#   -lr 0.1 -lbs 50 -ls 5 -nc 20 -gr 400 -m ResNet18 -did 0 -et 1 \
#   > ../outputs/lr-0.1_lbs-50_ls-5/FedALA/Cifar10_FedALA_ResNet18.out 2>&1 &
