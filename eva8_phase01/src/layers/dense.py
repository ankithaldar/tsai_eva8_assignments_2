#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Custom linear layer implementation with activation and initializer
'''


#imports
import torch
import torch.nn as nn

#   script imports
#imports

class Dense(nn.Module):
  '''
  Custom linear layer implementation with activation and initializer
  '''
  def __init__(self,
    in_features,
    out_features,
    bias=True,
    device=None,
    dtype=None,
    activation=None,
    initializer='xavier_uniform'
  ):
    super().__init__()

    self.dense = nn.Linear(
      in_features=in_features,
      out_features=out_features,
      bias=bias,
      device=device,
      dtype=dtype
    )
    self.__init_weights__(initializer)

    if activation is not None:
      self.activation = self.__get_activation__(activation)
    else:
      self.activation = None


  def __get_activation__(self, activation):
    act_dict = {
      'relu': nn.ReLU(),
      'tanh': nn.Tanh(),
      'sigmoid': nn.Sigmoid(),
      'softmax': nn.Softmax(dim=1),
      'log_softmax': nn.LogSoftmax(dim=1),
      'none': None
    }

    return act_dict.get(activation.lower(), None)


  def __init_weights__(self, initializer):
    if initializer.lower() == 'xavier_uniform':
      nn.init.xavier_uniform_(self.dense.weight)
    elif initializer.lower() == 'kaiming_uniform':
      nn.init.kaiming_uniform_(self.dense.weight, nonlinearity='relu')

    # self.dense.bias.data.fill_(0)


  def forward(self, input):
    if self.activation is None:
      return self.dense(input)
    else:
      return self.activation(self.dense(input))
# classes

def main():
  dense = Dense(120, 3)
  print(dense(torch.rand([120])))

if __name__ == '__main__':
  main()
