import os
import torch
import torchvision
import torchvision.transforms as transforms


def get_mean_sigma(device, dataset):
    if dataset == 'cifar10':
        mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((1, 3, 1, 1))
        sigma = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view((1, 3, 1, 1))
    elif dataset == "tinyimagenet":
        mean = torch.tensor([0.4802, 0.4481, 0.3975]).view(1, 3, 1, 1)
        sigma = torch.tensor([0.2302, 0.2265, 0.2262]).view(1, 3, 1, 1)
    else:
        mean = torch.FloatTensor([0.0]).view((1, 1, 1, 1))
        sigma = torch.FloatTensor([1.0]).view((1, 1, 1, 1))
    return mean.to(device), sigma.to(device)


def get_mnist():
    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    return train_set, test_set, 28, 1, 10


def get_fashion():
    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    return train_set, test_set, 28, 1, 10


def get_svhn():
    train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())
    return train_set, test_set, 32, 3, 10


def get_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 2, padding_mode='edge'),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    return train_set, test_set, 32, 3, 10


def get_tinyimagenet():
    train_set = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/train',
                                        transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(56, padding_mode='edge'),
                                            transforms.ToTensor(),
                                        ]))
    test_set = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/val',
                                        transform=transforms.Compose([
                                            transforms.CenterCrop(56),
                                            transforms.ToTensor(),
                                        ]))
    return train_set, test_set, 56, 3, 200


def get_loaders(args, num_workers:int=8, shuffle_test:bool=False):
    if args.dataset == 'cifar10':
        train_set, test_set, input_size, input_channels, n_class = get_cifar10()
    elif args.dataset == 'mnist':
        train_set, test_set, input_size, input_channels, n_class = get_mnist()
    elif args.dataset == 'fashion':
        train_set, test_set, input_size, input_channels, n_class = get_fashion()
    elif args.dataset == 'svhn':
        train_set, test_set, input_size, input_channels, n_class = get_svhn()
    elif args.dataset == "tinyimagenet":
        train_set, test_set, input_size, input_channels, n_class = get_tinyimagenet()
    else:
        raise NotImplementedError('Unknown dataset')

    if args.frac_valid is not None:
        n_valid = int(args.frac_valid * len(train_set))
        print('Using validation set of size %d!' % n_valid)
        train_set, val_set = torch.utils.data.random_split(train_set, [len(train_set) - n_valid, n_valid])
        val_set.transforms = transforms.ToTensor()
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.train_batch, shuffle=False, num_workers=8)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch, shuffle=True, num_workers=num_workers)
    test_batch = args.test_batch if args.grad_accu_batch is None else args.grad_accu_batch
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch, shuffle=shuffle_test, num_workers=num_workers)
    loaders = [train_loader, test_loader] if args.frac_valid is None else [train_loader, val_loader, test_loader]
    return loaders, len(train_set), input_size, input_channels, n_class

