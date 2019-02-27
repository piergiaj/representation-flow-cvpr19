import torch
import torch.utils.data as data_utl

import numpy as np
import random

import os
import lintel

import json



class MK(data_utl.Dataset):

    def __init__(self, split_file, root, mode='rgb', length=64, random=True, model='2d', size=112):
        with open(split_file, 'r') as f:
            self.data = json.load(f)
        self.vids = [k for k in self.data.keys()]

        if mode == 'flow':
            new_data = {}
            self.vids = ['flow'+v[3:] for v in self.vids]
            for v in self.data.keys():
                new_data['flow'+v[3:]] = self.data[v]
            self.data = new_data
        
        self.split_file = split_file
        self.root = root
        self.mode = mode
        self.model = model
        self.length = length
        self.random = random
        self.size = size

    def __getitem__(self, index):
        vid = self.vids[index]
        cls = self.data[vid]
        if not os.path.exists(os.path.join(self.root, vid)):
            if self.mode == 'flow' and self.model == '2d':
                return np.zeros((3, 20, self.size, self.size), dtype=np.float32), 0
            elif self.mode == 'flow' and self.model == '3d':
                return np.zeros((2, self.length, self.size, self.size), dtype=np.float32), 0
            

        with open(os.path.join(self.root, vid), 'rb') as f:
            enc_vid = f.read()

        
        df, w, h, _ = lintel.loadvid(enc_vid, should_random_seek=self.random, num_frames=self.length*2)
        df = np.frombuffer(df, dtype=np.uint8)

        if w < 128 or h < 128 or h > 512 or w > 512:
            df = np.zeros((self.length*2,128,128,3), dtype=np.uint8)
            w=h=128
            cls = 0

        # center crop
        if not self.random:
            i = int(round((h-self.size)/2.))
            j = int(round((w-self.size)/2.))
            df = np.reshape(df, newshape=(self.length*2, h, w, 3))[::2, i:-i, j:-j, :]
        else:
            th = self.size
            tw = self.size
            #print(h, th, h-th)
            i = random.randint(0, h - th) if h!=th else 0
            j = random.randint(0, w - tw) if w!=tw else 0
            df = np.reshape(df, newshape=(self.length*2, h, w, 3))[::2, i:i+th, j:j+tw, :]

            if random.random() < 0.5:
                df = np.flip(df, axis=2).copy()

        if self.mode == 'flow':
            #print(df[:,:,:,1:].mean())
            #exit()
            # only take the 2 channels corresponding to flow (x,y)
            df = df[:,:,:,1:]
            if self.model == '2d':
                # this should be redone...
                # stack 10 along channel axis
                df = np.asarray([df[:10],df[2:12],df[4:14]]) # gives 3x10xHxWx2
                df = df.transpose(0,1,4,2,3).reshape(3,20,self.size,self.size).transpose(0,2,3,1)
            
                
        df = 1-2*(df.astype(np.float32)/255)

        if self.model == '2d':
            # 2d -> return TxCxHxW
            return df.transpose([0,3,1,2]), cls
        # 3d -> return CxTxHxW
        return df.transpose([3,0,1,2]), cls
        
    def __len__(self):
        return len(self.data.keys())



if __name__ == '__main__':
    train = '/ssd/ajpiergi/minikinetics_train.json'
    val = '/ssd/ajpiergi/minikinetics_val.json'
    root = '/ssd/ajpiergi/minikinetics/'
    dataset_tr = MK(train, root, length=16, model='2d', mode='flow')
    print(dataset_tr[random.randint(0,1000)])

