import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random
import os
import time
import torchvision.transforms as transforms

from v7_main_model_functions import *
from v7_model_functions import *
from v7_1_get_data import *


if __name__ == '__main__':
    basic_path = './'

    variant_list = ['alpha', 'beta', 'gamma', 'delta', 'omicron']

    primer_length = 21

    if os.path.exists(f'{basic_path}vectorSize.csv'):
        vectorSize = pd.read_csv(f'{basic_path}vectorSize.csv', header=None).values[0][0]
    else:
        print('No vectorSize.csv file found')
        exit()

    method = 'Pooling' # 'Pooling', 'Top', 'Combination'
        
    if method == 'Pooling':
        forward_method = 'Pooling'
        forward_max_Pooling_window_size = 148
    elif method == 'Top':
        forward_method = 'Top'
        forward_top_number = 175
        forward_max_Pooling_window_size = int(vectorSize / forward_top_number) + 1
    elif method == 'Combination':
        forward_method = 'Combination'
        forward_max_Pooling_window_size = 500
        forward_top_in_window = 10


    for variant_Name in variant_list:
        print(f'\nNow running the --- {variant_Name} --- variant\n')
        path = f'{basic_path}{variant_Name}/forward/'

        if forward_method == 'Pooling':
            numberWindows = posPool_pytorch_cuda(path, vectorSize, forward_max_Pooling_window_size)
        elif forward_method == 'Top':
            numberWindows = posPool_top_pytorch_cuda(path, vectorSize, forward_max_Pooling_window_size)
        elif forward_method == 'Combination':
            numberWindows = posPool_combination_pytorch_cuda(path, vectorSize,
                                                             forward_max_Pooling_window_size,
                                                             forward_top_in_window)

        print(f'Forwaeed method: {forward_method}')
        print(f'Number of windows: {numberWindows}')

        createFeatVector_pytorch(path, numberWindows, vectorSize, primer_length)
        getFeature(basic_path, path, variant_Name, number = 20)
        sameFeature(path)
        calculateAppearance(basic_path, path, variant_Name)
        commonFeatureAppearance(path)
