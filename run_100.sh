#!/bin/bash
cd ./system || exit


#################### FedAvg ####################
nohup python -u main.py -algo FedAvg -data Cifar100-dir0.6-nc100 -ncl 100 \
  -nc 100 -jr 0.1 -lr 0.1 -lbs 50 -ls 5 -gr 2000 -m ResNet18 -did 0 \
  > ../outputs/nc100_jr0.1_lr-0.1_lbs-50/FedAvg/Cifar100_FedAvg_ResNet18_GN.out 2>&1 &

  nohup python -u main.py -algo FedAvg -data Cifar100-dir0.6-nc100 -ncl 100 \
  -nc 100 -jr 0.1 -lr 0.1 -lbs 50 -ls 5 -gr 2000 -m ResNet18 -did 0 \
  > ../outputs/nc100_jr0.1_lr-0.1_lbs-50/FedAvg/Cifar100-dir0.6_FedAvg_ResNet18_BN.out 2>&1 &



#################### FedDyn ####################
# nohup python -u main.py -algo FedDyn -data Cifar100-dir0.6-nc100 -ncl 100 \
#   -nc 100 -jr 0.1 -lr 0.1 -lbs 50 -ls 5 -gr 2000 -m ResNet18 -did 0 -al 0.1 \
#   > ../outputs/nc100_jr0.1_lr-0.1_lbs-50/FedDyn/Cifar100_FedDyn_al-0.1_ResNet18_GN.out 2>&1 &



#################### FedFGAC ####################
# nohup python -u main.py -algo FedFGAC -data Cifar100-dir0.6-nc100 -ncl 100 \
#   -nc 100 -jr 0.1 -lr 0.1 -lbs 50 -ls 5 -gr 2000 -m ResNet18 -did 0 -al 10 \
#   > ../outputs/nc100_jr0.1_lr-0.1_lbs-50/FedFGAC/Cifar100_FedFGAC_al10_Log_ResNet18_GN.out 2>&1 &



#################### FedCC ####################
# nohup python -u main.py -algo FedCC -data Cifar100-dir0.6-nc100 -ncl 100 \
#   -nc 100 -jr 0.1 -lr 0.1 -lbs 50 -ls 5 -gr 2000 -m ResNet18 -did 0 -al 10 \
#   > ../outputs/nc100_jr0.1_lr-0.1_lbs-50/FedCC/Cifar100_FedCC_-logL1_al10_Log_ResNet18_GN.out 2>&1 &



#################### FedFGAC_CC ####################
# nohup python -u main.py -algo FedFGAC_CC -data Cifar100-dir0.6-nc100 -ncl 100 \
#   -nc 100 -jr 0.1 -lr 0.1 -lbs 50 -ls 5 -gr 2000 -m ResNet18 -did 0 -al 10 \
#   > ../outputs/nc100_jr0.1_lr-0.1_lbs-50/FedFGAC_CC/Cifar100_FedFGAC_CC_-logL1_al10_Log_ResNet18.out 2>&1 &

# nohup python -u main.py -algo FedFGAC_CC -data Cifar100-dir0.6-nc100 -ncl 100 \
#   -nc 100 -jr 0.1 -lr 0.1 -lbs 50 -ls 5 -gr 2000 -m ResNet18 -did 0 -al 0.1 \
#   > ../outputs/nc100_jr0.1_lr-0.1_lbs-50/FedFGAC_CC/Cifar100_FedFGAC_CC_L1-100_al0.1_Log_ResNet18.out 2>&1 &
