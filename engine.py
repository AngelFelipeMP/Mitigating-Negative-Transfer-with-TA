import torch
import torch.nn as nn
from itertools import cycle

def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


def train_fn(data_loader, model, optimizer, device, scheduler, heads):
    model.train()
    finds = {head:{} for head in heads}
    loss = {head:None for head in heads}
    
    for i, (batch, head) in enumerate(zip(data_loader, cycle(heads)), start=0):
        
        if i % len(heads) == 0:
            optimizer.zero_grad()
        
        if i < len(heads):
            finds[head]['targets'] = []
            finds[head]['predictions'] = []
            finds[head]['loss'] = 0
            
        batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
        targets = batch["targets"]
        del batch["targets"]
        
        outputs = model(batch, head)
        loss[head] = loss_fn(outputs, targets)
        finds[head]['loss'] += (loss[head].cpu().detach().numpy().tolist()/(len(data_loader)/len(heads)))
        
        finds[head]['targets'].extend(targets.cpu().detach().numpy().tolist())
        _, predictions = torch.max(outputs, 1)
        finds[head]['predictions'].extend(predictions.cpu().detach().numpy().tolist())
        
        loss[head].backward()
        
        if (i + 1) % len(heads) == 0:
            optimizer.step()
            scheduler.step()
            
    return finds

def eval_fn(data_loader, model, device, heads):
    model.eval()
    finds = {head:{} for head in heads}
    
    with torch.no_grad():
        for i, (batch, head) in enumerate(zip(data_loader, cycle(heads)), start=0):
            
            if i < len(heads):
                finds[head]['targets'] = []
                finds[head]['predictions'] = []
                finds[head]['loss'] = 0
                
            batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
            targets = batch["targets"]
            del batch["targets"]

            outputs = model(batch, head)
            loss = loss_fn(outputs, targets)
            finds[head]['loss'] += (loss.cpu().detach().numpy().tolist()/(len(data_loader)/len(heads)))
            
            finds[head]['targets'].extend(targets.cpu().detach().numpy().tolist())
            _, predictions = torch.max(outputs, 1)
            finds[head]['predictions'].extend(predictions.cpu().detach().numpy().tolist())

    return finds
        
        

# def train_fn(data_loader, model, optimizer, device, scheduler, heads):
#     model.train()
#     finds = {}
    
#     for i, batch in enumerate(data_loader):
#         j = 0
        
#         optimizer.zero_grad()
        
#         while j < len(heads):
#             head = heads[j]
#             j =+ 1
            
#             if i < 1:
#                 finds[head]['targets'] = []
#                 finds[head]['predictions'] = []
#                 finds[head]['loss'] = 0
            
#             batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
#             targets = batch["targets"]
#             del batch["targets"]

#             outputs = model(batch, head)
#             loss = loss_fn(outputs, targets)
#             finds[head]['loss'] += loss.cpu().detach().numpy().tolist()/len(batch)
            
#             finds[head]['targets'].extend(targets.cpu().detach().numpy().tolist())
#             _, predictions = torch.max(outputs, 1)
#             finds[head]['predictions'].extend(predictions.cpu().detach().numpy().tolist())
            
#             loss.backward()
#         optimizer.step()
#         scheduler.step()

#     return finds


# def eval_fn(data_loader, model, device, heads):
#     model.eval()
#     finds = {}
    
#     with torch.no_grad():
#         for i, batch in enumerate(data_loader):
#             j = 0
#             while j < len(heads):
#                 head = heads.split('-')[j]
#                 j =+ 1
                
#                 if i <= 1:
#                     finds[head]['targets'] = []
#                     finds[head]['predictions'] = []
#                     finds[head]['loss'] = 0
                
#                 batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
#                 targets = batch["targets"]
#                 del batch["targets"]

#                 outputs = model(batch, head)
#                 loss = loss_fn(outputs, targets)
#                 finds[head]['loss'] += loss.cpu().detach().numpy().tolist()/len(batch)
                
#                 finds[head]['targets'].extend(targets.cpu().detach().numpy().tolist())
#                 _, predictions = torch.max(outputs, 1)
#                 finds[head]['predictions'].extend(predictions.cpu().detach().numpy().tolist())

#     return finds