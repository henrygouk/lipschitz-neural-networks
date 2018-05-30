#!/bin/bash

dub build -b release --single svhn.d
./svhn --datapath=/research/repository/hgrg1/svhn/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/svhn/vgg-none.model --logpath=logs/svhn/vgg-none.log --arch=vgg
./svhn --datapath=/research/repository/hgrg1/svhn/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/svhn/wrn-none.model --logpath=logs/svhn/wrn-none.log --arch=wrn
./svhn --datapath=/research/repository/hgrg1/svhn/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/svhn/vgg-dropout.model --logpath=logs/svhn/vgg-dropout.log --arch=vgg --dropout
./svhn --datapath=/research/repository/hgrg1/svhn/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/svhn/wrn-dropout.model --logpath=logs/svhn/wrn-dropout.log --arch=wrn --dropout
./svhn --datapath=/research/repository/hgrg1/svhn/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/svhn/vgg-sd.model --logpath=logs/svhn/vgg-sd.log --arch=vgg --spectral-decay=0.01
./svhn --datapath=/research/repository/hgrg1/svhn/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/svhn/wrn-sd.model --logpath=logs/svhn/wrn-sd.log --arch=wrn --spectral-decay=0.01
./svhn --datapath=/research/repository/hgrg1/svhn/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/svhn/vgg-batchnorm.model --logpath=logs/svhn/vgg-batchnorm.log --arch=vgg --batchnorm

