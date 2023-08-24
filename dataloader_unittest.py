import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
from math import gcd
# import fractions
def lcm(a,b): return abs(a * b)/gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
from matplotlib import pyplot as plt
for i, data in enumerate(dataset):
    if i > 1:
        break
    print(data['input_condition'], data['output_condition'])
    img1 = data['label'].numpy()[0, 0, :, :]
    img2 = data['image'].numpy()[0, 0, :, :]
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1)
    ax[0].set_title(data['input_condition'])
    ax[1].imshow(img2)
    ax[1].set_title(data['output_condition'])
    plt.show()

