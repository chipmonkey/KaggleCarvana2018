import os
os.environ['HOME'] = '/root'
#os.environ['PYTHONUNBUFFERED'] = '1'


#numerical libs
import math
import numpy as np
import random
import PIL
import cv2

import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg')


# torch libs
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import torch.backends.cudnn as cudnn



# std libs
import collections
import types
import numbers
import inspect
import shutil
#import pickle
import dill
from timeit import default_timer as timer   #ubuntu:  default_timer = time.time,  seconds
#import time

from datetime import datetime
import csv
import pandas as pd
import pickle
import glob
import sys
#from time import sleep
from distutils.dir_util import copy_tree
import zipfile
import zlib

import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics

from skimage import io
from sklearn.metrics import fbeta_score
'''
updating pytorch
    https://discuss.pytorch.org/t/updating-pytorch/309
    
    ./conda config --add channels soumith
    conda update pytorch torchvision
    conda install pytorch torchvision cuda80 -c soumith
    
check cudnn version

https://discuss.pytorch.org/t/cuda-cudnn-basics/6214
torch.backends.cudnn.version()
torch.cuda.is_available()

'''


#---------------------------------------------------------------------------------
print('@%s:  ' % os.path.basename(__file__))
if 1:
    SEED=235202
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    print ('\tset random seed')
    print ('\t\tSEED=%d'%SEED)

if 1:
    cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    print ('\tset cuda environment')


print('')

#---------------------------------------------------------------------------------