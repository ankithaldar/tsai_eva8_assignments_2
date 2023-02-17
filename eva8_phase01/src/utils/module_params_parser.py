#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Read parameters from YAML file.'''


#imports
from pathlib import Path

import yaml

try:
  from yaml import CLoader
except ImportError:
  from yaml import Loader as CLoader
#   script imports
#imports


def yml_reader(file_path: Path):
  if file_path.exists():
    with file_path.open('r') as f:
      return yaml.load(f, Loader=CLoader)
  else:
    print(f'Config file not found at {file_path.absolute()}.')
    raise FileNotFoundError


# classes
class ModelParamsDecoder:
  '''Read parameters from YAML file.'''

  def __init__(self, file_path:str, kernel_plat:str='colab'):
    self.file_path = Path(file_path)
    self.kernel_platform = kernel_plat
    # print(f'{self.file_path.absolute()}')

    self.decode_yml()

  def decode_yml(self):
    params = yml_reader(self.file_path)

    print('loading model parameters...')

    self.seed = params['seed']

    self.data_path = self.__get_create_path__(for_what='data')
    self.model_chkpt_path = self.__get_create_path__(for_what='model_chkpt_path')
    self.charts = self.__get_create_path__(for_what='charts')

    self.model_name = params['model_name']
    self.epochs = params['epochs']
    self.batch_size = params['batch_size']
    self.num_classes = params['num_classes']

    self.train_args = params['train_args']
    self.test_args = params['test_args']

    self.do_augment = params['do_augment']

    self.loss_function = params['loss_function']
    self.optimizer = params['optimizer']

    self.learning_rate = params['learning_rate']
    self.lr_scheduler = params['lr_scheduler']
    if self.lr_scheduler != 'None':
      self.lr_scheduler_args = params['lr_scheduler_args']

    self.metrics = params['metrics']

    self.callback_list = params['callbacks']

    self.logger_name = params['logger']
    self.logger_init_params = params['logger_init_params']

    self.norm_type = params['normalization']

    self.model_params = {
      'assignment': self.logger_init_params['assignment'],
      'model_name': self.model_name,
      'epochs': self.epochs,
      'batch_size': self.batch_size,
      'loss_function': self.loss_function,
      'optimizer': self.optimizer,
      'learning_rate': self.learning_rate,
      'metrics': self.metrics
    }

    print('model parameters loaded...')

  def __get_create_path__(self, for_what='data') -> Path:
    '''Get data path.'''

    def create_path(path: Path) -> None:
      '''Create if folder does not exist.'''
      if not path.exists():
        path.mkdir(parents=True)

    if self.kernel_platform == 'colab':
      pass

    path = Path(Path.cwd()/for_what)
    create_path(path)

    return path

# classes


# main
def main():
  ModelParamsDecoder('./config_params/mnist_lenet.yml')


# if main script
if __name__ == '__main__':
  main()
