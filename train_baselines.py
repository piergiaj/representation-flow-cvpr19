import os
import sys
import argparse
import inspect
import datetime
import json

import time

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-exp_name', type=str)
parser.add_argument('-model', type=str)
parser.add_argument('-batch_size', type=int, default=24)
parser.add_argument('-length', type=int, default=16)
parser.add_argument('-system', type=str, help='v100,k80,titanx,ultra')

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

#import models
import baseline_2d_resnets
import baseline_3d_resnets

device = torch.device('cuda')

##################
#
# Create model, dataset, and training setup
#
##################
if args.model == '2d':
    model = baseline_2d_resnets.resnet34(pretrained=True, mode=args.mode, dropout=0.8, num_classes=51, input_size=112)
else:
    model = baseline_3d_resnets.resnet50(pretrained=True, mode=args.mode, dropout=0.9, num_classes=51)
    
model = nn.DataParallel(model).to(device)
batch_size = args.batch_size


if args.system == 'titanx':
    train = '/data/ajpiergi/minikinetics_train.json'
    val = '/data/ajpiergi/minikinetics_val.json'
    root = '/data/ajpiergi/minikinetics/'
elif args.system == 'ultra':
    train = '/ssd/ajpiergi/minikinetics_train.json'
    val = '/ssd/ajpiergi/minikinetics_val.json'
    root = '/ssd/ajpiergi/minikinetics/'
elif args.system == 'v100':
    train = '/share/jproject/ajpiergi/minikinetics_train.json'
    val = '/share/jproject/ajpiergi/minikinetics_val.json'
    root = '/scratch_ssd/ajpiergi/minikinetics/'
elif args.system == 'k80':
    train = '/share/jproject/ajpiergi/minikinetics_train.json'
    val = '/share/jproject/ajpiergi/minikinetics_val.json'
    root = '/share/jproject/ajpiergi/minikinetics/'
elif args.system == 'hmdb':
    from hmdb_dataset import HMDB as DS
    dataseta = DS('data/hmdb/split1_train.txt', '/ssd/hmdb/', model=args.model, mode=args.mode, length=args.length)
    dl = torch.utils.data.DataLoader(dataseta, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    dataset = DS('data/hmdb/split1_test.txt', '/ssd/hmdb/', model=args.model, mode=args.mode, length=args.length, c2i=dataseta.class_to_id)
    vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dataloader = {'train':dl, 'val':vdl}


if args.system != 'hmdb':
    from minikinetics_dataset import MK
    dataset_tr = MK(train, root, length=args.length, model=args.model, mode=args.mode)
    dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    dataset = MK(val, root, length=args.length, model=args.model, mode=args.mode)
    vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dataloader = {'train':dl, 'val':vdl}


lr = 0.005
# this has worked somewaht well ~55% accuracy on MK
solver = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
#solver = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3, momentum=0.9, dampening=0.9)
lr_sched = optim.lr_scheduler.StepLR(solver, step_size=12, gamma=0.1)
#lr_sched = optim.lr_scheduler.ReduceLROnPlateau(solver, patience=10)


#################
#
# Setup logs, store model code
# hyper-parameters, etc...
#
#################
log_name = datetime.datetime.today().strftime('%m-%d-%H%M')+'-'+args.exp_name
log_path = os.path.join('logs/',log_name)
os.mkdir(log_path)

# deal with hyper-params...
with open(os.path.join(log_path,'params.json'), 'w') as out:
    hyper = vars(args)
    json.dump(hyper, out)
log = {'iterations':[], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[]}

    

###############
#
# Train the model and save everything
#
###############
num_epochs = 100
for epoch in range(num_epochs):

    for phase in ['train', 'val']:
        train = (phase=='train')
        if phase == 'train':
            model.train()
        else:
            model.eval()
            
        tloss = 0.
        acc = 0.
        tot = 0
        c = 0
        e=s=0

        with torch.set_grad_enabled(train):
            for vid, cls in dataloader[phase]:
                #if c%200 == 0:
                #    print('epoch',epoch,'iter',c)
                #s=time.time()
                #print('btw batch', (s-e)*1000)
                vid = vid.to(device)
                cls = cls.to(device)
                
                outputs = model(vid)
                
                pred = torch.max(outputs, dim=1)[1]
                corr = torch.sum((pred == cls).int())
                acc += corr.item()
                tot += vid.size(0)
                loss = F.cross_entropy(outputs, cls)
                #print(loss)
                
                if phase == 'train':
                    solver.zero_grad()
                    loss.backward()
                    solver.step()
                    log['iterations'].append(loss.item())
                    
                tloss += loss.item()
                c += 1
                #e=time.time()
                #print('batch',batch_size,'time',(e-s)*1000)
            
        if phase == 'train':
            log['epoch'].append(tloss/c)
            log['train_acc'].append(acc/tot)
            print('train loss',tloss/c, 'acc', acc/tot)
        else:
            log['validation'].append(tloss/c)
            log['val_acc'].append(acc/tot)
            print('val loss', tloss/c, 'acc', acc/tot)
            lr_sched.step(tloss/c)
    
    with open(os.path.join(log_path,'log.json'), 'w') as out:
        json.dump(log, out)
    torch.save(model.state_dict(), os.path.join(log_path, 'model.pt'))


    #lr_sched.step()
