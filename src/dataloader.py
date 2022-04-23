from torchvision import datasets
from torch.utils.data import DataLoader


def get_dataloader(batch_size, train_transform, val_transform, test_transform):
    # bs = 128
    # crop_size = 224
    train_set = datasets.ImageFolder(root='./data/processed/ip102_v1.1/images/train', transform = train_transform)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = datasets.ImageFolder(root='./data/processed/ip102_v1.1/images/val', transform=val_transform)
    validationloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.ImageFolder(root='./data/processed/ip102_v1.1/images/test', transform=test_transform)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return trainloader, validationloader, testloader