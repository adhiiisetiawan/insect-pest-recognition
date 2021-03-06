import torch
import torch.nn.functional as F

def sparse_loss(model, images, model_children):
    loss = 0
    values = images
    for i in range(len(model_children)):
        values = F.relu6((model_children[i](values)))
        loss += torch.mean(torch.abs(values))
    return loss