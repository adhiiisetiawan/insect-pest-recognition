import torch
from src.dataloader import get_dataloader
from src.data_augmentation import preprocess


train_transform, val_transfrom, test_transform = preprocess(crop_size=224)
train_dl, val_dl, test_dl = get_dataloader(128, train_transform, val_transfrom, test_transform)

# train_set.classes
print(train_dl.dataset.classes)
feature, target = next(iter(train_dl))
print(feature.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)