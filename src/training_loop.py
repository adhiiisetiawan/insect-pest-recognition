from tqdm import tqdm


def loop_function(mode, dataset, dataloader, model, criterion, optimizer, device, model_children):
    if mode == 'train':
        model.train()
    elif mode == 'val':
        model.eval()
          
    cost = correct = 0
    for feature, target in tqdm(dataloader, desc=mode.title()):
        feature, target = feature.to(device), target.to(device)

        if mode == 'train':
            batch = (feature, target)
            feature, target_a, target_b, lam = cutmix(batch, alpha=1.0)    
            output = model(feature) #feedforward
            
            cutmix_loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1 - lam) #hitung loss
            l1_loss = sparse_loss(model, feature, model_children)

            loss = cutmix_loss + 0.001 * l1_loss

            loss.backward() #backprop
            optimizer.step() #update weight
            optimizer.zero_grad()
        elif mode == 'val':
            output = model(feature) #feedforward
            cross_entropy_loss = criterion(output, target) #hitung loss
            l1_loss = sparse_loss(model, feature, model_children)
            # add the sparsity penalty
            loss = cross_entropy_loss + 0.001 * l1_loss
            
        cost += loss.item() * feature.shape[0]
        correct += (output.argmax(1) == target).sum().item()
    cost = cost / len(dataset)
    acc = correct / len(dataset)
    return cost, acc