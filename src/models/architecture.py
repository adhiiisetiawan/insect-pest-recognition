from torch import nn
from torchvision.models import resnet50

class InsectPestClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.freeze()
        self.resnet.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 102)
        )
        
    def forward(self, x):
        return self.resnet(x)
    
    def freeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
        
    def unfreeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = True