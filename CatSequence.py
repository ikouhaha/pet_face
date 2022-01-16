import PIL
import PIL.ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.ImageDraw
from Util import *
import tensorflow as tf

class CatSequence(tf.keras.utils.Sequence):

    def __init__(self, image_list,label_list, input_size, batch_size,
                 shuffle=False):
        
        
        #self.files_list = np.array(files_list)  # for advanced indexing
        self.image_list = image_list
        self.label_list = label_list
        self.input_size = input_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __len__(self):
        return int(np.ceil(len(self.image_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_i = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_fl = self.image_list[batch_i]
        image_list = self.image_list
        label_list = self.label_list
        
        
        return image_list, label_list

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_list))
        if self.shuffle:
            np.random.shuffle(self.indices)
