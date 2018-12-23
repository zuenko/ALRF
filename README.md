# Adaptive Mixture of Low-Rank Factorizations for Compact Neural Modeling

Implementation of "Adaptive Mixture of Low-Rank Factorizations for Compact Neural Modeling" ICLR 2019

#### Requires: 
- pytorch >= 0.4.1 
- torchvision >= 0.2.1
- tensorboard >= 1.12.0 
- tensorboardX >= 1.4.0 

## To do list:
- [x] Simple Network
- [x] MLP MNIST
- [x] MobileNet CIFAR
- [x] MobileNet CIFAR with Low-Rank Factorization
- [ ] Different datasets support (We need to change the architecture of network)
- [ ] Implement more models with Low-Rank Factorizations 
- [ ] Flops measurement 

# Contributors

- [Denis Zuenko](https://github.com/zuenko) has implemented MobileNet_CIFAR and MobileNet_CIFAR_LowRankShit.
- [Yuriy Gabuev](https://github.com/jurg96) has implemented the main idea of low-rank approximation and MLP.
- [Stanislav Tsepa](https://github.com/MrTsepa) has implemented LowRankLayer and MLP.
- [Van Khachatryan](https://github.com/vkhachatryan) has tested experiments and made a sum up.
- [Aleksandr Rubashevskii](https://github.com/rubaha96) has tested experiments and made a sum up.

# Reference
1. "[Adaptive Mixture of Low-Rank Factorizations for Compact Neural Modeling](https://openreview.net/forum?id=B1eHgu-Fim)" Ting Chen, Ji Lin, Tian Lin, Song Han, Chong Wang, Dengyong Zhou, ICLR 2019
