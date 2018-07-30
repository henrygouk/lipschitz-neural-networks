#!/bin/bash

datapath=/research/repository/hgrg1/mnist
modelpath=/research/repository/hgrg1/lipschitz-neural-networks/mnist
logpath=logs/mnist

dub build -b release --single mnist.d
./mnist --datapath=$datapath --modelpath=$modelpath/none.model --logpath=$logpath/none.log
./mnist --datapath=$datapath --modelpath=$modelpath/sd.model --logpath=$logpath/sd.log --spectral-decay=0.001
./mnist --datapath=$datapath --modelpath=$modelpath/batchnorm.model --logpath=$logpath/batchnorm.log --batchnorm
./mnist --datapath=$datapath --modelpath=$modelpath/dropout.model --logpath=$logpath/dropout.log --dropout
./mnist --datapath=$datapath --modelpath=$modelpath/l1.model --logpath=$logpath/l1.log --norm=1 --lambda=16
./mnist --datapath=$datapath --modelpath=$modelpath/l2.model --logpath=$logpath/l2.log --norm=2 --lambda=4
./mnist --datapath=$datapath --modelpath=$modelpath/linf.model --logpath=$logpath/linf.log --norm=inf --lambda=4
./mnist --datapath=$datapath --modelpath=$modelpath/l1-dropout.model --logpath=$logpath/l1-dropout.log --norm=1 --lambda=40 --dropout
./mnist --datapath=$datapath --modelpath=$modelpath/l2-dropout.model --logpath=$logpath/l2-dropout.log --norm=2 --lambda=8 --dropout
./mnist --datapath=$datapath --modelpath=$modelpath/linf-dropout.model --logpath=$logpath/linf-dropout.log --norm=inf --lambda=10 --dropout
./mnist --datapath=$datapath --modelpath=$modelpath/l1-batchnorm.model --logpath=$logpath/l1-batchnorm.log --batchnorm --norm=1 --lambda=4
./mnist --datapath=$datapath --modelpath=$modelpath/l2-batchnorm.model --logpath=$logpath/l2-batchnorm.log --batchnorm --norm=2 --lambda=0.2
./mnist --datapath=$datapath --modelpath=$modelpath/linf-batchnorm.model --logpath=$logpath/linf-batchnorm.log --batchnorm --norm=inf --lambda=0.2
./mnist --datapath=$datapath --modelpath=$modelpath/l1-batchnorm-dropout.model --logpath=$logpath/l1-batchnorm-dropout.log --batchnorm --norm=1 --lambda=3 --dropout
./mnist --datapath=$datapath --modelpath=$modelpath/l2-batchnorm-dropout.model --logpath=$logpath/l2-batchnorm-dropout.log --batchnorm --norm=2 --lambda=0.5 --dropout
./mnist --datapath=$datapath --modelpath=$modelpath/linf-batchnorm-dropout.model --logpath=$logpath/linf-batchnorm-dropout.log --batchnorm --norm=inf --lambda=0.3 --dropout

