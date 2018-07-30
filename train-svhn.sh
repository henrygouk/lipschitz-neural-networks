#!/bin/bash

datapath=/research/repository/hgrg1/svhn
modelpath=/research/repository/hgrg1/lipschitz-neural-networks/svhn
logpath=logs/svhn

dub build -b release --single svhn.d
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-none.model --logpath=$logpath/vgg-none.log --arch=vgg
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-dropout.model --logpath=$logpath/vgg-dropout.log --arch=vgg --dropout
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-sd.model --logpath=$logpath/vgg-sd.log --arch=vgg --spectral-decay=0.001
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-batchnorm.model --logpath=$logpath/vgg-batchnorm.log --arch=vgg --batchnorm
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-l1.model --logpath=$logpath/vgg-l1.log --arch=vgg --norm=1 --lambda=12
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-l2.model --logpath=$logpath/vgg-l2.log --arch=vgg --norm=2 --lambda=3
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-linf.model --logpath=$logpath/vgg-linf.log --arch=vgg --norm=inf --lambda=4
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-l1-dropout.model --logpath=$logpath/vgg-l1-dropout.log --arch=vgg --norm=1 --lambda=20 --dropout
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-l2-dropout.model --logpath=$logpath/vgg-l2-dropout.log --arch=vgg --norm=2 --lambda=5 --dropout
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-linf-dropout.model --logpath=$logpath/vgg-linf-dropout.log --arch=vgg --norm=inf --lambda=5 --dropout
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-l1-batchnorm.model --logpath=$logpath/vgg-l1-batchnorm.log --arch=vgg --norm=1 --batchnorm --lambda=7
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-l2-batchnorm.model --logpath=$logpath/vgg-l2-batchnorm.log --arch=vgg --norm=2 --batchnorm --lambda=1
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-linf-batchnorm.model --logpath=$logpath/vgg-linf-batchnorm.log --arch=vgg --norm=inf --batchnorm --lambda=1
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-l1-batchnorm-dropout.model --logpath=$logpath/vgg-l1-batchnorm-dropout.log --arch=vgg --norm=1 --batchnorm --lambda=7 --dropout
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-l2-batchnorm-dropout.model --logpath=$logpath/vgg-l2-batchnorm-dropout.log --arch=vgg --norm=2 --batchnorm --lambda=1 --dropout
./svhn --datapath=$datapath --modelpath=$modelpath/vgg-linf-batchnorm-dropout.model --logpath=$logpath/vgg-linf-batchnorm-dropout.log --arch=vgg --norm=inf --batchnorm --lambda=1 --dropout

./svhn --datapath=$datapath --modelpath=$modelpath/wrn-none.model --logpath=$logpath/wrn-none.log --arch=wrn
./svhn --datapath=$datapath --modelpath=$modelpath/wrn-dropout.model --logpath=$logpath/wrn-dropout.log --arch=wrn --dropout
./svhn --datapath=$datapath --modelpath=$modelpath/wrn-sd.model --logpath=$logpath/wrn-sd.log --arch=wrn --spectral-decay=0.001
./svhn --datapath=$datapath --modelpath=$modelpath/wrn-l1.model --logpath=$logpath/wrn-l1.log --arch=wrn --norm=1 --lambda=24
./svhn --datapath=$datapath --modelpath=$modelpath/wrn-l2.model --logpath=$logpath/wrn-l2.log --arch=wrn --norm=2 --lambda=5
./svhn --datapath=$datapath --modelpath=$modelpath/wrn-linf.model --logpath=$logpath/wrn-linf.log --arch=wrn --norm=inf --lambda=8
./svhn --datapath=$datapath --modelpath=$modelpath/wrn-l1-dropout.model --logpath=$logpath/wrn-l1-dropout.log --arch=wrn --norm=1 --lambda=24 --dropout
./svhn --datapath=$datapath --modelpath=$modelpath/wrn-l2-dropout.model --logpath=$logpath/wrn-l2-dropout.log --arch=wrn --norm=2 --lambda=5 --dropout
./svhn --datapath=$datapath --modelpath=$modelpath/wrn-linf-dropout.model --logpath=$logpath/wrn-linf-dropout.log --arch=wrn --norm=inf --lambda=8 --dropout
