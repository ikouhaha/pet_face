import random
import cv2, os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
from pathlib import Path
import datetime
import sys
import tensorflow as tf
import platform


import tensorflow.keras as keras
import tensorflow as tf
import shutil

import datetime
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from math import atan2, degrees
# from minetorch.miner import Miner
# from minetorch.metrics import MultiClassesClassificationMetricWithLogic
# from minetorch.plugins.noise_detector import NoiseSampleDetector
# from minetorch.spreadsheet import GoogleSheet
# from torchvision import datasets, transforms
#from minetorch.spreadsheet.GoogleSheet
