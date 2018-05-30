Lipschitz Continuous Neural Networks
====================================

This repo contains the code used to run the experiments in Gouk et al. (2018). The code uses [dopt](https://github.com/henrygouk/dopt/), a deep learning framework written in D.

Example
-------

The following command will train a wide residual network on the CIFAR-10 dataset:

```
./cifar10.d --datapath ~/Datasets/cifar10/ --norm=inf --lambda=3 --arch=wrn
```

References
----------

Gouk, H., Frank, E., Pfahringer, B., & Cree, M. (2018). Regularisation of Neural Networks by Enforcing Lipschitz Continuity. arXiv preprint arXiv:1804.04368.