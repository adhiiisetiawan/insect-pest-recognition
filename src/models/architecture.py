from torch import nn
from torchvision.models import densenet169


class InsectPestClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = densenet169(pretrained=True)
        self.freeze()
        self.densenet.classifier = nn.Linear(1664, 102)
        
    def forward(self, x):
        return self.densenet(x)
    
    def freeze(self):
        for param in self.densenet.parameters():
            param.requires_grad = False
        
    def unfreeze(self):
        for param in self.densenet.parameters():
            param.requires_grad = True
