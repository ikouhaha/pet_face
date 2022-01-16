import tensorflow.keras as keras
import numpy as np
import os

all_files_loc = "datapsycho/imglake/population/train/image_files/"
all_files = os.listdir(all_files_loc)

image_label_map = {
        "image_file_{}.npy".format(i+1): "label_file_{}.npy".format(i+1)
        for i in range(int(len(all_files)/2))}

class DataGenerator(keras.utils.Sequence):

    def __init__(self, file_list):
        """Constructor can be expanded,
           with batch size, dimentation etc.
        """
        self.file_list = file_list
        self.on_epoch_end()

    def __len__(self):
      'Take all batches in each iteration'
      return int(len(self.file_list))

    def __getitem__(self, index):
      'Get next batch'
      # Generate indexes of the batch
      indexes = self.indexes[index:(index+1)]

      # single file
      file_list_temp = [self.file_list[k] for k in indexes]

      # Set of X_train and y_train
      X, y = self.__data_generation(file_list_temp)

      return X, y

    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.file_list))

    def __data_generation(self, file_list_temp):
      'Generates data containing batch_size samples'
      data_loc = "datapsycho/imglake/population/train/image_files/"
      # Generate data
      for ID in file_list_temp:
          x_file_path = os.path.join(data_loc, ID)
          y_file_path = os.path.join(data_loc, image_label_map.get(ID))

          # Store sample
          X = np.load(x_file_path)

          # Store class
          y = np.load(y_file_path)

      return X, y