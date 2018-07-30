#!/bin/bash

datapath=/research/repository/hgrg1/cifar10
modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar10
logpath=logs/cifar10

dub build -b release --single cifar10.d

./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-none.model --logpath=$logpath/vgg-none.log --arch=vgg
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-dropout.model --logpath=$logpath/vgg-dropout.log --arch=vgg --dropout
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-sd.model --logpath=$logpath/vgg-sd.log --arch=vgg --spectral-decay=0.01
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-batchnorm.model --logpath=$logpath/vgg-batchnorm.log --arch=vgg --batchnorm
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-l1.model --logpath=$logpath/vgg-l1.log --arch=vgg --norm=1 --lambda=28
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-linf.model --logpath=$logpath/vgg-linf.log --arch=vgg --norm=inf --lambda=7
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-l2.model --logpath=$logpath/vgg-l2.log --arch=vgg --norm=2 --lambda=4
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-dropout-l1.model --logpath=$logpath/vgg-dropout-l1.log --arch=vgg --norm=1 --lambda=40 --dropout
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-dropout-linf.model --logpath=$logpath/vgg-dropout-linf.log --arch=vgg --norm=inf --lambda=8 --dropout
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-dropout-l2.model --logpath=$logpath/vgg-dropout-l2.log --arch=vgg --norm=2 --lambda=4 --dropout
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-batchnorm-l1.model --logpath=$logpath/vgg-batchnorm-l1.log --arch=vgg --norm=1 --lambda=20 --batchnorm
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-batchnorm-l2.model --logpath=$logpath/vgg-batchnorm-l2.log --arch=vgg --norm=2 --lambda=1.5 --batchnorm
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-batchnorm-linf.model --logpath=$logpath/vgg-batchnorm-linf.log --arch=vgg --norm=inf --lambda=3 --batchnorm
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-dropout-batchnorm-l1.model --logpath=$logpath/vgg-dropout-batchnorm-l1.log --arch=vgg --norm=1 --lambda=24 --batchnorm --dropout
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-dropout-batchnorm-l2.model --logpath=$logpath/vgg-dropout-batchnorm-l2.log --arch=vgg --norm=2 --lambda=2 --batchnorm --dropout
./cifar10 --datapath=$datapath --modelpath=$modelpath/vgg-dropout-batchnorm-linf.model --logpath=$logpath/vgg-dropout-batchnorm-linf.log --arch=vgg --norm=inf --lambda=4 --batchnorm --dropout

./cifar10 --datapath=$datapath --modelpath=$modelpath/wrn-none.model --logpath=$logpath/wrn-none.log --arch=wrn
./cifar10 --datapath=$datapath --modelpath=$modelpath/wrn-dropout.model --logpath=$logpath/wrn-dropout.log --arch=wrn --dropout
./cifar10 --datapath=$datapath --modelpath=$modelpath/wrn-sd.model --logpath=$logpath/wrn-sd.log --arch=wrn --spectral-decay=0.01
./cifar10 --datapath=$datapath --modelpath=$modelpath/wrn-l1.model --logpath=$logpath/wrn-l1.log --arch=wrn --norm=1 --lambda=56
./cifar10 --datapath=$datapath --modelpath=$modelpath/wrn-l2.model --logpath=$logpath/wrn-l2.log --arch=wrn --norm=2 --lambda=8
./cifar10 --datapath=$datapath --modelpath=$modelpath/wrn-linf.model --logpath=$logpath/wrn-linf.log --arch=wrn --norm=inf --lambda=14
./cifar10 --datapath=$datapath --modelpath=$modelpath/wrn-l1-dropout.model --logpath=$logpath/wrn-l1-dropout.log --arch=wrn --norm=1 --lambda=56 --dropout
./cifar10 --datapath=$datapath --modelpath=$modelpath/wrn-l2-dropout.model --logpath=$logpath/wrn-l2-dropout.log --arch=wrn --norm=2 --lambda=8 --dropout
./cifar10 --datapath=$datapath --modelpath=$modelpath/wrn-linf-dropout.model --logpath=$logpath/wrn-linf-dropout.log --arch=wrn --norm=inf --lambda=14 --dropout
