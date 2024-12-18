#!/bin/bash
cd ./system || exit


#################### FedAvg ####################
# nohup python -u main.py -algo FedAvg -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -nc 20 -gr 400 -m CNN -did 0 \
#   > ../outputs/FedAvg/Cifar10_FedAvg_CNN.out 2>&1 &

# nohup python -u main.py -algo FedAvg -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -nc 20 -gr 400 -m CNN -did 0 \
#   > ../outputs/FedAvg/Cifar100_FedAvg_CNN.out 2>&1 &



#################### FedFGAC ####################
# nohup python -u main.py -algo FedFGAC -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/FedFGAC/Cifar10_FedFGAC_update_CNN.out 2>&1 &

# nohup python -u main.py -algo FedFGAC -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/FedFGAC/Cifar100_FedFGAC_update_CNN.out 2>&1 &



#################### FedLC ####################
# nohup python -u main.py -algo FedLC -data Cifar10 -ncl 10 \
#   -tau 1 -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/FedLC/Cifar10_FedLC_CNN.out 2>&1 &

# nohup python -u main.py -algo FedLC -data Cifar100 -ncl 100 \
#   -tau 1 -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/FedLC/Cifar100_FedLC_CNN.out 2>&1 &

# nohup python -u main.py -algo FedLC_New -data Cifar10 -ncl 10 \
#   -tau 1 -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/FedLC/Cifar10_FedLC_New_CNN.out 2>&1 &

# nohup python -u main.py -algo FedLC_New -data Cifar100 -ncl 100 \
#   -tau 1 -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/FedLC/Cifar100_FedLC_New_CNN.out 2>&1 &



#################### FedNTD ####################
# nohup python -u main.py -algo FedNTD -data Cifar10 -ncl 10 \
#   -bt 1 -tau 1 -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/FedNTD/Cifar10_FedNTD_CNN.out 2>&1 &

# nohup python -u main.py -algo FedNTD -data Cifar100 -ncl 100 \
#   -bt 1 -tau 1 -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/FedNTD/Cifar100_FedNTD_CNN.out 2>&1 &



#################### FedFGAC_NTD ####################
nohup python -u main.py -algo FedFGAC_NTD -data Cifar10 -ncl 10 \
  -bt 0.1 -tau 1 -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
  > ../outputs/FedFGAC_NTD/Cifar10_FedFGAC_NTD_bt0.1_CNN.out 2>&1 &

nohup python -u main.py -algo FedFGAC_NTD -data Cifar100 -ncl 100 \
  -bt 0.1 -tau 1 -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
  > ../outputs/FedFGAC_NTD/Cifar100_FedFGAC_NTD_bt0.1_CNN.out 2>&1 &
