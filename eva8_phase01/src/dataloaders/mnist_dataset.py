#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Dataloader for downloading and batching data'''


#imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

#   script imports
#imports


# classes
class MNISTRandomDataLoader(Dataset):
  '''Dataloader for downloading and batching data'''

  def __init__(self, root:str='', train=False):
    # super(DataLoader, self).__init__()
    self.dataset = self.__download_dataset__(root=root, train=train)

    self.labels = self.dataset.targets
    self.image = self.dataset.data

    # generate random number
    self.random_num = torch.randint(
      low=0,
      high=10,
      size=(len(self.dataset),),
      dtype=torch.float
    )

  def __download_dataset__(self, root:str='', train=False):
    '''Downloads the dataset'''

    return torchvision.datasets.MNIST(
      root=root,
      train=train,
      download=True,
      transform=transforms.ToTensor()
    )

  def __len__(self):
    return len(self.image)

  def __getitem__(self, index:int):
    '''Gets the sample'''

    image = self.image[index]
    label = self.labels[index]
    rand_int = self.random_num[index]

    return (image/255.).reshape([1, 28, 28]), label, rand_int, label + rand_int

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
