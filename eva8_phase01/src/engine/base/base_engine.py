#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Base Engine to train models'''


#imports
import torch
import torch.nn as nn
import pytorch_lightning as pl
#   script imports
from utils.module_params_parser import ModelParamsDecoder
#imports


# classes
class BaseEngine(pl.LightningModule):
  '''Base Engine to train models'''

  def __init__(self, hparams:ModelParamsDecoder):
    super().__init__()

    self.hparams = hparams

    self._init_train_dataloader()
    self._init_test_dataloader()

    self._init_model()
    self._init_loss_function()
    self._init_metrics()
    self._init_callbacks()

    self.setup()

  def _init_train_dataloader(self):
    self.train_ds = None

  def _init_test_dataloader(self):
    self.test_ds = None

  def _init_valid_dataloader(self):
    self.valid_ds = None



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
