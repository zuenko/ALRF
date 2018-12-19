import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms


class FlattenBatch(nn.Module):
    def __init__(self):
        super(FlattenBatch, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class LowRankLayer(nn.Module):
    def __init__(self, input_size, output_size=2 ** 8, d=8, K=2, pi_size=28):
        """
        d: rank of decompositions
        K: number of pairs (U(k), V(k))
        pi_size: size of pi function hidden layer
        """
        super(LowRankLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.d = d
        self.K = K
        self.pi_size = pi_size

        self.Vs1 = [nn.Parameter(torch.randn(self.input_size, self.d, requires_grad=True)) for k in range(self.K)]
        self.Us1 = [nn.Parameter(torch.randn(self.d, self.output_size, requires_grad=True)) for k in range(self.K)]
        for k in range(self.K):
            self.register_parameter('U{}'.format(k), self.Us1[k])
            self.register_parameter('V{}'.format(k), self.Vs1[k])

        self.W_pi = nn.Parameter(torch.randn(self.pi_size, self.K), requires_grad=True)

    def forward(self, x):
        # x has shape (n_samples, channel_count, input_size)
        if len(x.shape) > 3:
            x = x.view(x.shape[0], x.shape[1], -1)
        pool_size = x.shape[2] // self.pi_size
        x_pooled = F.avg_pool1d(x, pool_size)  # (n_samples, channel_count, pi_size)
        pi = torch.sigmoid(x_pooled.matmul(self.W_pi))  # (n_samples, channel_count, K)
        Wx = [(x.matmul(self.Vs1[k])).matmul(self.Us1[k]) for k in range(self.K)]
        Wx = torch.stack(Wx, dim=2)  # (n_samples, channel_count, K, output_size)
        Wx = (pi.view(*pi.shape, 1) * Wx).sum(dim=2)  # (n_samples, channel_count, output_size)

        return Wx


if __name__ == '__main__':
    batch_size = 128
    batch_size_test = 1000
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size_test, shuffle=True)


    def train(model, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))


    def _test(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


    lr = 0.01
    momentum = 0.9
    n_epochs = 100

    model = nn.Sequential(
        LowRankLayer(input_size=28 * 28, output_size=2 ** 8, d=2, K=2),
        nn.Sigmoid(),
        FlattenBatch(),
        nn.Linear(in_features=2**8, out_features=10),
        nn.Softmax(dim=1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        train(model, train_loader, optimizer, epoch)
        _test(model, test_loader)
