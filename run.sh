#!/bin/bash
cd ./system || exit


#################### FedAvg ####################
# nohup python -u main.py -algo FedAvg -data Cifar10 -ncl 10 \
#   -lr 0.003 -lbs 32 -ls 3 -nc 20 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.01_lbs-32_ls-3/FedAvg/Cifar10_FedAvg_CNN.out 2>&1 &

# nohup python -u main.py -algo FedAvg -data Cifar10 -ncl 10 \
#   -lr 0.1 -lbs 50 -ls 5 -nc 20 -gr 400 -m ResNet18 -did 0 \
#   > ../outputs/lr-0.1_lbs-50_ls-5/FedAvg/Cifar10_FedAvg_ResNet18.out 2>&1 &

# nohup python -u main.py -algo FedAvg -data Cifar100 -ncl 100 \
#   -lr 0.003 -lbs 32 -ls 3 -nc 20 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.01_lbs-32_ls-3/FedAvg/Cifar100_FedAvg_CNN.out 2>&1 &


# nohup python -u main.py -algo FedAvg -data Cifar100 -ncl 100 \
#   -lr 0.1 -lbs 50 -ls 5 -gr 2000 -m ResNet18 -did 0 \
#   > ../outputs/lr-0.1_lbs-50_ls-5/FedAvg/Cifar100_FedAvg_wd0.001_ResNet18-BN.out 2>&1 &



#################### FedFGAC ####################
# nohup python -u main.py -algo FedFGAC -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC/Cifar10_FedFGAC_Log_CNN_test.out 2>&1 &

# nohup python -u main.py -algo FedFGAC -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC/Cifar100_FedFGAC_Log_CNN.out 2>&1 &

# nohup python -u main.py -algo FedFGAC -data Cifar10 -ncl 10 \
#   -lr 0.01 -lbs 32 -ls 3 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.01_lbs-32_ls-3/FedFGAC/Cifar10_FedFGAC_wd1e-5_CNN.out 2>&1 &

# nohup python -u main.py -algo FedFGAC -data Cifar100 -ncl 100 \
#   -lr 0.01 -lbs 32 -ls 3 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.01_lbs-32_ls-3/FedFGAC/Cifar100_FedFGAC_wd1e-5_CNN.out 2>&1 &

# python -u main.py -algo FedFGAC -data Cifar10 -ncl 10 \
#   -lr 0.003 -lbs 32 -ls 3 -gr 400 -m CNN -did 0 \

# python -u main.py -algo FedFGAC -data Cifar100 -ncl 100 \
#   -lr 0.003 -lbs 32 -ls 3 -gr 400 -m CNN -did 0 \



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
# nohup python -u main.py -algo FedFGAC_NTD -data Cifar10 -ncl 10 \
#   -bt 0.5 -tau 1 -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC_NTD/Cifar10_FedFGAC_Log_NTD_bt0.5_CNN.out 2>&1 &

# nohup python -u main.py -algo FedFGAC_NTD -data Cifar100 -ncl 100 \
#   -bt 0.5 -tau 1 -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC_NTD/Cifar100_FedFGAC_Log_NTD_bt0.5_CNN.out 2>&1 &



#################### FedFGAC_MCD ####################
# nohup python -u main.py -algo FedFGAC_MCD -data Cifar10 -ncl 10 \
#   -bt 0.1 -tau 1 -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC_MCD/Cifar10_FedFGAC_Log_MCD_bt0.1_CNN.out 2>&1 &

# nohup python -u main.py -algo FedFGAC_MCD -data Cifar100 -ncl 100 \
#   -bt 0.1 -tau 1 -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC_MCD/Cifar100_FedFGAC_Log_MCD_bt0.1_CNN.out 2>&1 &




#################### FedAvg_Frozen ####################
# nohup python -u main.py -algo FedAvg_Frozen -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedAvg_Frozen/Cifar10_FedAvg_Frozen_Log_CNN_test.out 2>&1 &

# nohup python -u main.py -algo FedAvg_Frozen -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedAvg_Frozen/Cifar100_FedAvg_Frozen_Log_CNN.out 2>&1 &



#################### FedFGAC_Frozen ####################
# nohup python -u main.py -algo FedFGAC_Frozen -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC_Frozen/Cifar10_FedFGAC_Frozen_Log_CNN_test.out 2>&1 &

# nohup python -u main.py -algo FedFGAC_Frozen -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC_Frozen/Cifar100_FedFGAC_Frozen_Log_CNN.out 2>&1 &



#################### FedFGAC ####################
# nohup python -u main.py -algo FedFGAC -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC/Cifar10_FedFGAC_LSR0.001_Log_CNN.out 2>&1 &

# nohup python -u main.py -algo FedFGAC -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC/Cifar100_FedFGAC_LSR0.1_Log_CNN.out 2>&1 &



#################### FedFGAC_CC ####################
# nohup python -u main.py -algo FedFGAC_CC -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -al 1 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC_CC/Cifar10_FedFGAC_CC-L1-2000_al1_Log_CNN.out 2>&1 &

# nohup python -u main.py -algo FedFGAC_CC -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -al 100 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC_CC/Cifar10_FedFGAC_CC_-logL2_al100_Log_CNN.out 2>&1 &

# nohup python -u main.py -algo FedFGAC_CC -data Cifar10 -ncl 10 \
#   -lr 0.1 -lbs 50 -ls 5 -gr 400 -m ResNet18 -did 0 -al 1 \
#   > ../outputs/lr-0.1_lbs-50_ls-5/FedFGAC_CC/Cifar10_FedFGAC_CC_-logL1_al1_Log_ResNet18-GN.out 2>&1 &

# nohup python -u main.py -algo FedFGAC_CC -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -al 1 \
#   > ../outputs/lr-0.005_lbs-10/FedFGAC_CC/Cifar100_FedFGAC_CC_L1-100_al1_Log_CNN.out 2>&1 &


# nohup python -u main.py -algo FedFGAC_CC -data Cifar100 -ncl 100 \
#   -lr 0.1 -lbs 50 -ls 5 -gr 2000 -m ResNet18 -did 0 -al 10 \
#   > ../outputs/lr-0.1_lbs-50_ls-5/FedFGAC_CC/Cifar100_FedFGAC_CC_-logL1_wd0.001_al10_Log_ResNet18-BN.out 2>&1 &



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
# nohup python -u main.py -algo FedDyn -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -al 0.01 \
#   > ../outputs/lr-0.005_lbs-10/FedDyn/Cifar10_FedDyn_al-0.01_CNN.out 2>&1 &

# nohup python -u main.py -algo FedDyn -data Cifar10 -ncl 10 \
#   -lr 0.1 -lbs 50 -ls 5 -gr 400 -m ResNet18 -did 0 -al 0.01 \
#   > ../outputs/lr-0.1_lbs-50_ls-5/FedDyn/Cifar10_FedDyn_al-0.01_ResNet18-GN.out 2>&1 &

# nohup python -u main.py -algo FedDyn -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -al 0.1 \
#   > ../outputs/lr-0.005_lbs-10/FedDyn/Cifar100_FedDyn_al-0.1_CNN.out 2>&1 &


nohup python -u main.py -algo FedDyn -data Cifar100 -ncl 100 \
  -lr 0.1 -lbs 50 -ls 5 -gr 2000 -m ResNet18 -did 0 -al 0.01 \
  > ../outputs/lr-0.1_lbs-50_ls-5/FedDyn/Cifar100_FedDyn-BN_wd0.00001_al-0.01_ResNet18-BN.out 2>&1 &



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



#################### FedTGCR ####################
# nohup python -u main.py -algo FedTGCR -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -se 100 -lam 0.1 \
#   > ../outputs/lr-0.005_lbs-10/FedTGCR/Cifar10_FedTGCR_G_lam0.1_CNN.out 2>&1 &

# nohup python -u main.py -algo FedTGCR -data Cifar100 -ncl 100 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -se 100 -lam 100 \
#   > ../outputs/lr-0.005_lbs-10/FedTGCR/Cifar100_FedTGCR_G_lam100_CNN.out 2>&1 &



#################### FedTGP ####################
# nohup python -u main.py -algo FedTGP -data Cifar10 -ncl 10 \
#   -lr 0.005 -lbs 10 -ls 1 -gr 400 -m CNN -did 0 -se 100 -lam 0.1 -mart 100 \
#   > ../outputs/lr-0.005_lbs-10/FedTGP/Cifar10_FedTGP_lam0.1.out 2>&1 &










#################### FedALA ####################
# nohup python -u main.py -algo FedALA -data Cifar10 -ncl 10 \
#   -lr 0.1 -lbs 50 -ls 5 -nc 20 -gr 400 -m ResNet18 -did 0 -et 1 \
#   > ../outputs/lr-0.1_lbs-50_ls-5/FedALA/Cifar10_FedALA_ResNet18.out 2>&1 &
