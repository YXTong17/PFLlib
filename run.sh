#!/bin/bash
cd ./system || exit

#################### FedAvg ####################
nohup python -u main.py -algo FedAvg -data Cifar10 -ncl 10 \
  -lr 0.01 -lbs 32 -ls 1 -nc 20 -m CNN -did 0 \
  > ../outputs/FedAvg/Cifar10_FedAvg_CNN.out 2>&1 &

nohup python -u main.py -algo FedAvg -data Cifar100 -ncl 100 \
  -lr 0.01 -lbs 32 -ls 1 -nc 20 -m CNN -did 0 \
  > ../outputs/FedAvg/Cifar100_FedAvg_CNN.out 2>&1 &



#################### FedFGAC ####################
nohup python -u main.py -algo FedFGAC -data Cifar10 -ncl 10 \
  -lr 0.01 -lbs 32 -ls 1 -m CNN -did 0 \
  > ../outputs/FedFGAC/Cifar10_FedFGAC_CNN.out 2>&1 &

nohup python -u main.py -algo FedFGAC -data Cifar100 -ncl 100 \
  -lr 0.01 -lbs 32 -ls 1 -m CNN -did 0 \
  > ../outputs/FedFGAC/Cifar100_FedFGAC_CNN.out 2>&1 &
