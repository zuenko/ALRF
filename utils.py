import torch.utils.data
from torchvision import datasets, transforms
import torch.utils.data as data

def build_dataset(dataset='MNIST', dataset_dir='./data', batch_size=100):
    dataset_ = {
        'MNIST': datasets.MNIST,
        'CIFAR10': datasets.CIFAR10,
        'STL10': datasets.STL10
    }[dataset]
    
    transform = {
        'MNIST': transforms.ToTensor(),
        'CIFAR10': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'STL10':transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    }[dataset]
    
    train_dataset = dataset_(root=dataset_dir,
                             train=True,
                             transform=transform,
                             download=True)

    train_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    test_dataset = dataset_(root=dataset_dir,
                             train=False,
                             transform=transform,
                             download=True)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)
    
    return train_loader, test_loader

def flops_to_string(flops):
    if flops // 10**9 > 0:
        return str(round(flops / 10.**9, 2)) + ' GMac'
    elif flops // 10**6 > 0:
        return str(round(flops / 10.**6, 2)) + ' MMac'
    elif flops // 10**3 > 0:
        return str(round(flops / 10.**3, 2)) + ' KMac'
    return str(flops) + ' Mac'

def params_to_string(params_num):
    if params_num // 10 ** 6 > 0:
        return str(round(params_num / 10 ** 6, 2)) + 'M'
    elif params_num // 10 ** 3:
        return str(round(params_num / 10 ** 3, 2)) + 'k'

    return str(params_num)