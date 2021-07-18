import os
import sys
import argparse
import inspect
import datetime
import json

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

#import models
import flow_2p1d_resnets

# device = torch.device('cuda')

##################
#
# Create model, dataset, and training setup
#
##################


class Model:
    def __init__(self, device):
        self.device = device

    def train(self, args):
        model = flow_2p1d_resnets.resnet50(pretrained=False, mode=args.mode, n_iter=args.niter,
                                           learnable=eval(args.learnable), num_classes=400)

        model = nn.DataParallel(model).to(self.device)
        batch_size = args.batch_size

        if args.system == 'hmdb':
            from hmdb_dataset import HMDB as DS
            dataseta = DS('data/hmdb/split1_train.txt', '/ssd/hmdb/', model=args.model, mode=args.mode, length=args.length)
            dl = torch.utils.data.DataLoader(dataseta, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

            dataset = DS('data/hmdb/split1_test.txt', '/ssd/hmdb/', model=args.model, mode=args.mode, length=args.length, c2i=dataseta.class_to_id)
            vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
            dataloader = {'train':dl, 'val':vdl}

        if args.system == 'minikinetics':
            train = 'data/kinetics/minikinetics_train.json'
            val = 'data/kinetics/minikinetics_val.json'
            root = '/ssd/kinetics/'
            from minikinetics_dataset import MK
            dataset_tr = MK(train, root, length=args.length, model=args.model, mode=args.mode)
            dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

            dataset = MK(val, root, length=args.length, model=args.model, mode=args.mode)
            vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
            dataloader = {'train':dl, 'val':vdl}

        if args.system == 'kinetics':
            train = 'data/kinetics/kinetics_train.json'
            val = 'data/kinetics/kinetics_val.json'
            root = '/ssd/kinetics/'
            from minikinetics_dataset import MK
            dataset_tr = MK(train, root, length=args.length, model=args.model, mode=args.mode)
            dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

            dataset = MK(val, root, length=args.length, model=args.model, mode=args.mode)
            vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
            dataloader = {'train':dl, 'val':vdl}


        # scale lr for flow layer
        params = model.parameters()
        params = [p for p in params]
        other = []
        print(len(params))
        ln = eval(args.learnable)
        if ln[0] == 1:
            other += [p for p in params if (p.sum() == model.module.flow_layer.img_grad.sum()).all() and p.size() == model.module.flow_layer.img_grad.size()]
            other += [p for p in params if (p.sum() == model.module.flow_layer.img_grad2.sum()).all() and p.size() == model.module.flow_layer.img_grad2.size()]
            params = [p for p in params if (p.sum() != model.module.flow_layer.img_grad.sum()).all() or p.size() != model.module.flow_layer.img_grad.size()]
            params = [p for p in params if (p.sum() != model.module.flow_layer.img_grad2.sum()).all() or p.size() != model.module.flow_layer.img_grad2.size()]

        if ln[1] == 1:
            other += [p for p in params if (p.sum() == model.module.flow_layer.f_grad.sum()).all() and p.size() == model.module.flow_layer.f_grad.size()]
            other += [p for p in params if (p.sum() == model.module.flow_layer.f_grad2.sum()).all() and p.size() == model.module.flow_layer.f_grad2.size()]
            params = [p for p in params if (p.sum() != model.module.flow_layer.f_grad.sum()).all() or p.size() != model.module.flow_layer.f_grad.size()]
            params = [p for p in params if (p.sum() != model.module.flow_layer.f_grad2.sum()).all() or p.size() != model.module.flow_layer.f_grad2.size()]

        if ln[2] == 1:
            other += [p for p in params if (p.sum() == model.module.flow_layer.t.sum()).all() and p.size() == model.module.flow_layer.t.size()]
            params = [p for p in params if (p.sum() != model.module.flow_layer.t.sum()).all() or p.size() != model.module.flow_layer.t.size()]

        if ln[3] == 1:
            other += [p for p in params if (p.sum() == model.module.flow_layer.l.sum()).all() and p.size() == model.module.flow_layer.l.size()]
            params = [p for p in params if (p.sum() != model.module.flow_layer.l.sum()).all() or p.size() != model.module.flow_layer.l.size()]

        if ln[4] == 1:
            other += [p for p in params if (p.sum() == model.module.flow_layer.a.sum()).all() and p.size() == model.module.flow_layer.a.size()]
            params = [p for p in params if (p.sum() != model.module.flow_layer.a.sum()).all() or p.size() != model.module.flow_layer.a.size()]



        #print([p for p in model.parameters() if (p == model.module.flow_layer.t).all()])
        #print(other)
        print(len(params), len(other))
        #exit()

        lr = 0.01
        solver = optim.SGD([{'params':params}, {'params':other, 'lr':0.01*lr}], lr=lr, weight_decay=1e-6, momentum=0.9)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(solver, patience=7)


        #################
        #
        # Setup logs, store model code
        # hyper-parameters, etc...
        #
        #################
        log_name = datetime.datetime.today().strftime('%m-%d-%H%M')+'-'+args.exp_name
        log_path = os.path.join('logs/',log_name)
        os.mkdir(log_path)
        os.system('cp * logs/'+log_name+'/')

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
        num_epochs = 60
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
                        if c%200 == 0:
                            print('epoch',epoch,'iter',c)
                        #s=time.time()
                        #print('btw batch', (s-e)*1000)
                        vid = vid.to(self.device)
                        cls = cls.to(self.device)

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
            torch.save(model.state_dict(), os.path.join(log_path, 'hmdb_flow-of-flow_2p1d.pt'))


            #lr_sched.step()
