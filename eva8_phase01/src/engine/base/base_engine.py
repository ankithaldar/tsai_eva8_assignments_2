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

    self.setup()

  def setup(self):
    self._init_optimizer()

  # Lightning module functions
  def configure_optimizers(self):
    if self.hparams.lr_scheduler == 'none':
      return self.optimizer()
    else:
      return (
        {
          'optimizer': self.optimizer(),
          'lr_scheduler': self._init_lr_scheduler()
        }
      )

  def
  # Lightning module functions

  def _init_optimizer(self):
    if self.hparams.optimizer == 'SGD':
      self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.hparams.learning_rate,
        momentum=0
      )
    elif self.hparams.optimizer == 'Adam':
      self.optimizer = torch.optim.Adam(
        self.model.parameters(),
        lr=self.hparams.learning_rate
      )

  def _init_lr_scheduler(self):
    if self.hparams.lr_scheduler == 'StepLR':
      return {
        'scheduler': torch.optim.lr_scheduler.StepLR(
          self.optimizer,
          **self.hparams.lr_scheduler_args
        )
      }


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
