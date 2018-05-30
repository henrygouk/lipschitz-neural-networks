#!/bin/bash

dub build -b release --single cifar100.d
./cifar100 --datapath=/research/repository/hgrg1/cifar100/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar100/vgg-none.model --logpath=logs/cifar100/vgg-none.log --arch=vgg
./cifar100 --datapath=/research/repository/hgrg1/cifar100/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar100/wrn-none.model --logpath=logs/cifar100/wrn-none.log --arch=wrn
./cifar100 --datapath=/research/repository/hgrg1/cifar100/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar100/vgg-dropout.model --logpath=logs/cifar100/vgg-dropout.log --arch=vgg --dropout
./cifar100 --datapath=/research/repository/hgrg1/cifar100/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar100/wrn-dropout.model --logpath=logs/cifar100/wrn-dropout.log --arch=wrn --dropout
./cifar100 --datapath=/research/repository/hgrg1/cifar100/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar100/vgg-sd.model --logpath=logs/cifar100/vgg-sd.log --arch=vgg --spectral-decay=0.01
./cifar100 --datapath=/research/repository/hgrg1/cifar100/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar100/wrn-sd.model --logpath=logs/cifar100/wrn-sd.log --arch=wrn --spectral-decay=0.01
./cifar100 --datapath=/research/repository/hgrg1/cifar100/ --modelpath=/research/repository/hgrg1/lipschitz-neural-networks/cifar100/vgg-batchnorm.model --logpath=logs/cifar100/vgg-batchnorm.log --arch=vgg --batchnorm

