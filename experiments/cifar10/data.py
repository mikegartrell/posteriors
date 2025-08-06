import numpy as np
import torch
from torchvision import transforms, datasets

# Prepare and load data
def prepare_data(args, data_root='./data', train_data_aug=True):
    '''
        Setups:
            -Original train and val data are merged, then split into (train, val)
            -Original test data is used as a test split as it is
    '''

    # data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # official train+val split
    train_set = datasets.CIFAR10(
        root=data_root, train=True,
        transform=transform_train if train_data_aug else transform_test,
        download=True
    )
    if args.val_heldout > 0:
        val_set = datasets.CIFAR10(
            root=data_root, train=True,
            transform=transform_test, download=True
        )
        val_size = int(args.val_heldout * len(train_set))
        train_size = len(train_set) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_set, _ = torch.utils.data.random_split(train_set, [train_size, val_size], generator=generator)
        val_set = torch.utils.data.Subset(val_set, np.setdiff1d(np.arange(len(val_set)), train_set.indices))
    test_set = datasets.CIFAR10(
        root=data_root, train=False,
        transform=transform_test, download=True
    )

    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size=args.batch_size, shuffle=train_data_aug, pin_memory=args.use_cuda, num_workers=4
    )
    if args.val_heldout > 0:
        val_loader = torch.utils.data.DataLoader(val_set,
            batch_size=args.batch_size, shuffle=False, pin_memory=args.use_cuda, num_workers=4
        )
    else:
        val_loader = None
    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=args.batch_size, shuffle=False, pin_memory=args.use_cuda, num_workers=4
    )
    args.num_classes = 10
    
    return train_loader, val_loader, test_loader, len(train_set)