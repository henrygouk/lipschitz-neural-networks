#!/bin/bash

dub build -b release --single cifar10.d
./cifar10 --datapath=/research/repository/hgrg1/cifar10/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar10/vgg-none.model --logpath=logs/cifar10/vgg-none.log --arch=vgg
./cifar10 --datapath=/research/repository/hgrg1/cifar10/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar10/wrn-none.model --logpath=logs/cifar10/wrn-none.log --arch=wrn
./cifar10 --datapath=/research/repository/hgrg1/cifar10/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar10/vgg-dropout.model --logpath=logs/cifar10/vgg-dropout.log --arch=vgg --dropout
./cifar10 --datapath=/research/repository/hgrg1/cifar10/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar10/wrn-dropout.model --logpath=logs/cifar10/wrn-dropout.log --arch=wrn --dropout
./cifar10 --datapath=/research/repository/hgrg1/cifar10/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar10/vgg-sd.model --logpath=logs/cifar10/vgg-sd.log --arch=vgg --spectral-decay=0.01
./cifar10 --datapath=/research/repository/hgrg1/cifar10/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar10/wrn-sd.model --logpath=logs/cifar10/wrn-sd.log --arch=wrn --spectral-decay=0.01
./cifar10 --datapath=/research/repository/hgrg1/cifar10/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar10/vgg-batchnorm.model --logpath=logs/cifar10/vgg-batchnorm.log --arch=vgg --batchnorm

