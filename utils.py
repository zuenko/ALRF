import torch.utils.data
from torchvision import datasets, transforms

def load_mnist(batch_size = 128, batch_size_test = 1000):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist_data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size_test, shuffle=True)
    
    return train_loader, test_loader