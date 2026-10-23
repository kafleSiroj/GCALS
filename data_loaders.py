import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import ssl

class Loader:
    def __init__(self, train_batch=None, test_batch=None):
        self.train_batch = train_batch
        self.test_batch = test_batch

    @staticmethod
    def transform(mean: tuple, std: tuple):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def get_cifar10(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        torchvision.datasets.CIFAR10.url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"

        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        transform = self.transform(mean, std)
        
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_batch, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.test_batch, shuffle=False)

        return train_loader, test_loader

    def get_cifar100(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        torchvision.datasets.CIFAR100.url = "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz"

        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        transform = self.transform(mean, std)
        
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_batch, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.test_batch, shuffle=False)

        return train_loader, test_loader

    def get_tiny_imagenet(self):
        from tinyimagenet import TinyImageNet

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        transform = self.transform(mean, std)

        data_dir = Path("./tinyimagenet_data")
        
        train_dataset = TinyImageNet(data_dir, split="train", transform=transform)
        test_dataset = TinyImageNet(data_dir, split="val", transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_batch, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.test_batch, shuffle=False)

        return train_loader, test_loader


    