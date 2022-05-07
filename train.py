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
train_dl, val_dl, test_dl = get_dataloader(batch_size=8,
                                           train_transform=train_transform,
                                           val_transfrom=val_transfrom,
                                           test_transform=test_transform)

model = InsectPestClassifier().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

summary(model, (3, 224, 224))

# large-scale-pest-recognition
wandb.init(project='test-ip102', name="DenseNet169", reinit=True)
wandb.watch(model, log='all')

epochs = 300
train_cost, val_cost = [], []
train_acc, val_acc = [], []
for i in range(epochs):
    cost_train, acc_train = loop_function('train', train_dl.dataset, train_dl, model, criterion, optimizer, device)
    train_cost.append(cost_train)
    train_acc.append(acc_train)
    with torch.no_grad():
        cost_val, acc_val = loop_function('val', val_dl.dataset, val_dl, model, criterion, optimizer, device)
        val_cost.append(cost_val)
        val_acc.append(acc_val)
      
    wandb.log({"train_cost":train_cost[-1], "val_cost":val_cost[-1], "train_acc": train_acc[-1], "val_acc":val_acc[-1]})
    print(f"\rEpoch: {i+1}/{epochs} | train_cost: {train_cost[-1]:.4f} | val_cost: {val_cost[-1]:.4f} | "
          f"train_acc: {train_acc[-1]:.4f} | val_acc: {val_acc[-1]:.4f}")

print("Finished training")
wandb.finish()

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "./output/checkpoint1.pth")
torch.save(model.state_dict(), "./output/mobilenetv3_cutmix_sparse_dlr.pth")