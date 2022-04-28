import torch
import wandb

from torch import nn, optim
from torchsummary import summary
from src.dataset import get_dataloader
from src.models import InsectPestClassifier
from torch.optim.lr_scheduler import StepLR
from src.augmentation import preprocess
from src.training_loop import loop_function


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_transform, val_transfrom, test_transform = preprocess(crop_size=224)
train_dl, val_dl, test_dl = get_dataloader(128, train_transform, val_transfrom, test_transform)

model = InsectPestClassifier().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
model_children = list(model.children())

summary(model, (3, 224, 224))


wandb.init(project='large-scale-pest-recognition', name="MobileNetV3-large_cutmix_sparse_dlr", reinit=True)
wandb.watch(model, log='all')

epochs = 150
train_cost, val_cost = [], []
train_acc, val_acc = [], []
for i in range(epochs):
    cost_train, acc_train = loop_function('train', train_dl.dataset, train_dl, model, criterion, optimizer, device, model_children)
    train_cost.append(cost_train)
    train_acc.append(acc_train)
    with torch.no_grad():
        cost_val, acc_val = loop_function('val', val_dl.dataset, val_dl, model, criterion, optimizer, device, model_children)
        val_cost.append(cost_val)
        val_acc.append(acc_val)
    scheduler.step()
      
    wandb.log({"train_cost":train_cost[-1], "val_cost":val_cost[-1], "train_acc": train_acc[-1], "val_acc":val_acc[-1]})
    print(f"\rEpoch: {i+1}/{epochs} | train_cost: {train_cost[-1]:.4f} | val_cost: {val_cost[-1]:.4f} | "
          f"train_acc: {train_acc[-1]:.4f} | val_acc: {val_acc[-1]:.4f} | Learning rate: {scheduler.get_last_lr()}")

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler":scheduler.state_dict(),
}, "./output/checkpoint1-mobilenetv3-large.pth")


# Fine Tuning
model.unfreeze()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

summary(model, (3, 224, 224))

ft_epochs = 150
train_cost, val_cost = [], []
train_acc, val_acc = [], []
for i in range(ft_epochs):
    cost_train, acc_train = loop_function('train', train_dl.dataset, train_dl, model, criterion, optimizer, device, model_children)
    train_cost.append(cost_train)
    train_acc.append(acc_train)
    with torch.no_grad():
        cost_val, acc_val = loop_function('val', val_dl.dataset, val_dl, model, criterion, optimizer, device, model_children)
        val_cost.append(cost_val)
        val_acc.append(acc_val)
    scheduler.step()
      
    wandb.log({"train_cost":train_cost[-1], "val_cost":val_cost[-1], "train_acc": train_acc[-1], "val_acc":val_acc[-1]})
    print(f"\rEpoch: {i+1}/{epochs} | train_cost: {train_cost[-1]:.4f} | val_cost: {val_cost[-1]:.4f} | "
          f"train_acc: {train_acc[-1]:.4f} | val_acc: {val_acc[-1]:.4f} | Learning rate: {scheduler.get_last_lr()}")

print("Finished training")
wandb.finish()

torch.save(model.state_dict(), "./output/mobilenetv3-large_cutmix_sparse_dlr.pth")