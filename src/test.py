import torch
from dataset import get_dataloader
from augmentation import preprocess
from models import InsectPestClassifier
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_transform, val_transfrom, test_transform = preprocess(crop_size=224)
train_dl, val_dl, test_dl = get_dataloader(128, train_transform, val_transfrom, test_transform)

filename = "/home/adhi/output-model/mobilenetv2_cutmix_sparse_dlr.pth"

model = InsectPestClassifier().to(device)
model.load_state_dict(torch.load(filename))


num_correct = 0
num_samples = 0
model.eval()

with torch.no_grad():
    for x, y in tqdm(test_dl, desc="Testing"):
        x = x.to(device=device)
        y = y.to(device=device)
        
        scores = model(x)
        predictions = scores.argmax(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)
    
    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 