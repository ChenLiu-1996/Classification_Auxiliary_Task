import torch
import torchvision
import random
from easydict import EasyDict

RANDOM_SEED = 0
TRAIN_VAL_RATIO = 0.95 # train : (train + val)

def load_dataset(name, batch_size, num_workers=2):
    """Load train, validation and test data."""
    
    if name == 'cifar-10':
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            # For the numbers below, see https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        eval_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # For the numbers below, see https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        img_dim = [32, 32, 3] # 32 x 32 images with 3 channels (R, G, B)

        train_dataset = torchvision.datasets.CIFAR10(
            './data/cifar-10', train=True, download=True, transform=train_transform)
        validation_dataset = torchvision.datasets.CIFAR10(
            './data/cifar-10', train=True, download=True, transform=eval_transform)
        test_dataset = torchvision.datasets.CIFAR10(
            './data/cifar-10', train=False, download=True, transform=eval_transform)

        # Further split the train dataset into train and validation sets.
        num_train_val_samples = int(len(train_dataset))
        num_train_samples = int(len(train_dataset) * TRAIN_VAL_RATIO)

        random.seed(RANDOM_SEED)
        train_indices = random.sample(range(num_train_val_samples), num_train_samples)
        validation_indices = list(set(range(num_train_val_samples)) - set(train_indices))

        train_dataset = torch.utils.data.dataset.Subset(train_dataset, train_indices)
        validation_dataset = torch.utils.data.dataset.Subset(validation_dataset, validation_indices)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return EasyDict(img_dim=img_dim, train=train_loader, validation=validation_loader, test=test_loader)
    else:
        pass