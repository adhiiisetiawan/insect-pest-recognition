import torch

from torch import nn, optim
from src.dataloader import get_dataloader
from src.model import InsectPestClassifier
from torch.optim.lr_scheduler import StepLR
from src.data_augmentation import preprocess


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_transform, val_transfrom, test_transform = preprocess(crop_size=224)
train_dl, val_dl, test_dl = get_dataloader(128, train_transform, val_transfrom, test_transform)

model = InsectPestClassifier().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
model_children = list(model.children())