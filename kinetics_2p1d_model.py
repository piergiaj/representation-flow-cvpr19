from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import rep_flow_layer as rf


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys

class SamePadding(nn.Module):

  def __init__(self, kernel_size, stride):
    super(SamePadding, self).__init__()
    self.kernel_size = kernel_size
    self.stride = stride

  def compute_pad(self, dim, s):
    if s % self.stride[dim] == 0:
      return max(self.kernel_size[dim] - self.stride[dim], 0)
    else:
      return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)
    
  def forward(self, x):
    # compute 'same' padding
    (batch, channel, t, h, w) = x.size()
    #print t,h,w
    out_t = np.ceil(float(t) / float(self.stride[0]))
    out_h = np.ceil(float(h) / float(self.stride[1]))
    out_w = np.ceil(float(w) / float(self.stride[2]))
    #print out_t, out_h, out_w
    pad_t = self.compute_pad(0, t)
    pad_h = self.compute_pad(1, h)
    pad_w = self.compute_pad(2, w)
    #print pad_t, pad_h, pad_w
    
    pad_t_f = pad_t // 2
    pad_t_b = pad_t - pad_t_f
    pad_h_f = pad_h // 2
    pad_h_b = pad_h - pad_h_f
    pad_w_f = pad_w // 2
    pad_w_b = pad_w - pad_w_f
    
    pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
    #print x.size()
    #print pad
    x = F.pad(x, pad)
    return x


class Bottleneck3D(nn.Module):
  
  def __init__(self, inputs, filters, is_training, strides,
               use_projection=False, T=3, data_format='channels_last', non_local=False):
    """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block.
  """
    super(Bottleneck3D, self).__init__()
            
    df = 'NDHWC' if data_format == 'channels_last' else 'NCDHW'
    self.shortcut = None
    if use_projection:
      # Projection shortcut only in first block within a group. Bottleneck blocks
      # end with 4 times the number of filters.
      filters_out = 4 * filters
      self.shortcut = nn.Sequential(SamePadding((1,1,1),(1,strides,strides)),
                                    nn.Conv3d(
                                      inputs, filters_out, kernel_size=1, stride=(1,strides,strides), bias=False, padding=0),
                                    nn.BatchNorm3d(filters_out),
                                    nn.BatchNorm3d(filters_out)) # there are two, due to old models having it. To load weights, 2 batch norms are needed here...
    
    self.layers = nn.Sequential(SamePadding((T,1,1), (1,1,1)),
                                nn.Conv3d(inputs, filters, kernel_size=(T,1,1), stride=1, padding=(0,0,0), bias=False), #1
                                nn.BatchNorm3d(filters), #2
                                nn.ReLU(),
                                SamePadding((1,3,3),(1,strides,strides)),
                                nn.Conv3d(filters, filters, kernel_size=(1,3,3), stride=(1,strides,strides), bias=False, padding=0), #5
                                nn.BatchNorm3d(filters),#6
                                nn.ReLU(),
                                nn.Conv3d(filters, 4*filters, kernel_size=1, stride=1, bias=False, padding=0),#8
                                nn.BatchNorm3d(4*filters))#9
    

  def forward(self, x):
    #print('block', x.size())
    if self.shortcut:
      res = self.shortcut(x)
    else:
      res = x
    #print('b2',x.size())
    return F.relu(self.layers(x) + res)


  
class Block3D(nn.Module):
  def __init__(self, inputs, filters, block_fn, blocks, strides, is_training, name,
                   data_format='channels_last', non_local=0):
    """Creates one group of blocks for the ResNet model.
    
  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
    is_training: `bool` for whether the model is training.
    name: `str`name for the Tensor output of the block layer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block layer.
    """
    super(Block3D, self).__init__()

    self.blocks = nn.ModuleList()
    # Only the first block per block_group uses projection shortcut and strides.
    self.blocks.append(Bottleneck3D(inputs, filters, is_training, strides,
                                    use_projection=True, data_format=data_format))
    inputs = filters * 4
    T = 3
    for i in range(1, blocks):
      self.blocks.append(Bottleneck3D(inputs, filters, is_training, 1, T=T,
                                      data_format=data_format, non_local=0))
      # only use 1 3D conv per 2 residual blocks (per Non-local NN paper)
      T = (3 if T==1 else 1)
    

  def forward(self, x):
    for block in self.blocks:
      x = block(x)
    return x

class ResNet3D(nn.Module):
  
  def __init__(self, block_fn, layers, num_classes,
               data_format='channels_last', non_local=[], rep_flow=[],
               dropout_keep_prob=0.5):
    """Generator for ResNet v1 models.

  Args:
    block_fn: `function` for the block to use within the model. Either
        `residual_block` or `bottleneck_block`.
    layers: list of 4 `int`s denoting the number of blocks to include in each
      of the 4 block groups. Each group consists of blocks that take inputs of
      the same resolution.
    num_classes: `int` number of possible classes for image classification.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the ResNet model.
    """
    super(ResNet3D, self).__init__()
    is_training = False # no effect in pytorch

    # TODO: RF Layer!
    #self.rep_flow = rep_flow_layer.rep_flow(inputs, rep_flow[0], is_training, bottleneck=1, data_format=data_format, use_last_conv=False)

    """Creation of the model graph."""
    self.stem = nn.Conv3d(
      3, 64, kernel_size=7, bias=False, stride=2)
    
    self.bn1 = nn.BatchNorm3d(64, eps=0.001, momentum=0.01)
    self.relu = nn.ReLU(inplace=True)
    self.pad = SamePadding((3,3,3),(2,2,2))
    self.maxpool = nn.MaxPool3d(kernel_size=3,
                                stride=2, padding=0)

    self.rep_flow = rf.FlowLayer(512)

    # res 2
    inputs = 64
    self.res2 = Block3D(
      inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
      strides=1, is_training=is_training, name='block_group1',
      data_format=data_format, non_local=non_local[0])
    
    # res 3
    inputs = 64*4
    self.res3 = Block3D(
      inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
      strides=2, is_training=is_training, name='block_group2',
      data_format=data_format, non_local=non_local[1])

    # res 4
    inputs = 128*4
    self.res4 = Block3D(
      inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
      strides=2, is_training=is_training, name='block_group3',
      data_format=data_format, non_local=non_local[2])

    # res 5
    inputs = 256*4
    self.res5 = Block3D(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_group4',
        data_format=data_format, non_local=non_local[3])

    self.dropout = nn.Dropout(0.5)
    self.classify = nn.Conv3d(512*4, num_classes, kernel_size=1, stride=1)

  def forward(self, x):
    print(x.size())
    x = self.stem(x)
    x = self.bn1(x)
    x = F.relu(x)
    print(x.size())
    x = self.maxpool(self.pad(x))
    print(x.size())
    x = self.res2(x)
    x = self.res3(x)

    x = self.rep_flow(x)

    x = self.res4(x)
    x = self.res5(x)
    x = x.mean(3).mean(3).unsqueeze(3).unsqueeze(3) # spatial average
    x = self.dropout(x)
    x = self.classify(x)
    x = x.mean(2) # temporal average
    return x


def resnet_3d_v1(resnet_depth, num_classes, data_format='channels_last', is_3d=True, non_local=[0,0,0,0], rep_flow=[0,0,0,0,0]):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': None, 'layers': [2, 2, 2, 2]},
      34: {'block': None, 'layers': [3, 4, 6, 3]},
      50: {'block': None, 'layers': [3, 4, 6, 3]},
      101: {'block': None, 'layers': [3, 4, 23, 3]},
      152: {'block': None, 'layers': [3, 8, 36, 3]},
      200: {'block': None, 'layers': [3, 24, 36, 3]}
  }

  if resnet_depth not in model_params:
    raise ValueError('Not a valid resnet_depth:', resnet_depth)

  params = model_params[resnet_depth]
  return ResNet3D(
    params['block'], params['layers'], num_classes, data_format, non_local, rep_flow)



if __name__ == '__main__':
  net = resnet_3d_v1(50, 400)

  net.load_state_dict(torch.load('models/rep_flow.pt'))
  
  data = np.load('data/v_CricketShot_g04_c01_rgb.npy')
  data = np.transpose(data, (0,4,1,2,3))
  tmp = torch.from_numpy(data)
  net.eval()
  with torch.no_grad():
    predclass = torch.argmax(net(tmp).squeeze())

  with open('data/label_map.txt') as f:
    classes = f.readlines()

  print(classes[predclass])
