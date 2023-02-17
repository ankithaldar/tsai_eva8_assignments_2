#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''LeNet for the model'''


#imports
import torch
import torch.nn as nn

#   script imports
from layers.conv2d import Conv2D
from layers.dense import Dense

#imports


# classes
class LeNet(nn.Module):
  '''LeNet for the model'''

  def __init__(self, num_classes: int):
    super(LeNet, self).__init__()

    # LAYERS FOR IMAGE RECOGNITION
    self.conv_01_1 = Conv2D(in_channels=1, out_channels=6, kernel_size=(3, 3))
    self.conv_01_2 = Conv2D(in_channels=6, out_channels=6, kernel_size=(3, 3))

    self.pool_01 = nn.AvgPool2d(kernel_size=(2, 2))

    self.conv_02_1 = Conv2D(in_channels= 6, out_channels=16, kernel_size=(3, 3))
    self.conv_02_2 = Conv2D(in_channels=16, out_channels=16, kernel_size=(3, 3))

    self.pool_02 = nn.AvgPool2d(kernel_size=(2, 2))

    self.flatten = nn.Flatten()

    self.fc_1 = Dense(in_features=256, out_features=120, activation='sigmoid')
    self.fc_2 = Dense(in_features=120, out_features=84 , activation='sigmoid')
    self.fc_3 = Dense(in_features=84 , out_features=num_classes, activation='softmax')

    # LAYERS FOR ADDITION
    self.fc_a_1 = Dense(in_features=2 , out_features=10, activation='sigmoid')
    self.fc_a_2 = Dense(in_features=10, out_features=6 , activation='sigmoid')
    self.fc_a_3 = Dense(in_features=6 , out_features=2*num_classes-1, activation='none')

  def forward(self, input, num):
    x = input

    x = self.conv_01_1(x)
    x = self.conv_01_2(x)
    x = self.pool_01(x)

    x = self.conv_02_1(x)
    x = self.conv_02_2(x)
    x = self.pool_02(x)

    x = self.flatten(x)

    x = self.fc_1(x)
    x = self.fc_2(x)
    x = self.fc_3(x)

    x1 = torch.stack((x.argmax(dim=1), num), dim=-1)

    x1 = self.fc_a_1(x1)
    x1 = self.fc_a_2(x1)
    x1 = self.fc_a_3(x1)

    return x, x1
# classes


# functions
def function_name():
  pass
# functions


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  main()
