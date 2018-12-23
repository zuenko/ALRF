import torch
import torch.utils.data
from torchvision import datasets, transforms
import torch.utils.data as data

import os

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
        'STL10': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)