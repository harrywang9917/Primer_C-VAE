import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sympy.physics.units import length
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import shutil
import random
import os
import time
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from v7_main_model_functions import *
from v7_model_functions import *
from v7_1_get_data import *
from v7_2_forward_model import *


if __name__ == '__main__':
    
    basic_path = './'

    variant_list = ['alpha', 'beta', 'gamma', 'delta', 'omicron']
    
    primer_length = 21

    learning_rate = 1e-4
    num_epochs = 50
    beta = 0.1
    lambda_class = 10.0

    generate_reverse_number = 10
    train_model_number_reverse = 1000
    total_number_reverse = generate_reverse_number + train_model_number_reverse

    reverse_batchSize = 32
    keep_prob = 0.5
    
    method = 'Pooling'
    
    if method == 'Pooling':
        reverse_method = 'Pooling'
        reverse_max_Pooling_window_size = 148
    elif method == 'Top':
        reverse_method = 'Top'
        reverse_top_number = 300
    elif method == 'Combination':
        reverse_method = 'Combination'
        reverse_max_Pooling_window_size = 500
        reverse_top_in_window = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n\n")
    
    for variant_Name in variant_list:
        old_path = basic_path + variant_Name + '/forward/' + 'Appearance_DataFrame.csv'
        new_path = basic_path + variant_Name + '/reverse/forward_primer/' + 'Appearance_DataFrame.csv'
        shutil.copyfile(old_path, new_path)
        
    for variant_Name in variant_list:
        print(f'Generating reverse primers for {variant_Name} variant')
        path = f'{basic_path}{variant_Name}/reverse/'
        
        get_forward_primers(path)
        CG_content_check(path, min_CG=0.4, max_CG=0.6)
        
        seq_file = basic_path + 'seq_and_seqName/{}_GISAID_{}.csv'.format('seq', variant_Name)
        sequence_full = pd.read_csv(seq_file, header=None).values.ravel()
        
        primers = pd.read_csv(path + 'forward_primer_CG_check.csv', header=None).values.ravel()

        # check if the reverse primer is already generated
        files = glob.glob(os.path.join(path, 'result', '*'))
        exist_primer_set = {os.path.splitext(os.path.basename(file))[0] for file in files 
                            if not file.endswith('/temp')}
        temp_list = [primer for primer in primers if primer not in exist_primer_set]
        # update primers
        primers = temp_list
        
        for forward_primer in tqdm(primers, desc="Processing Froward Primers"):
            # get sequence after forward primer
            sequence_original = get_after_primer_data(path, forward_primer, sequence_full,
                                                      total_number_reverse)
            # check if the forward primer is available as reverse primer
            exist_primer_check(path, forward_primer, sequence_original)
            
            vectorSize = max(len(seq) for seq in sequence_original)
            
            if reverse_method == 'Top':
                reverse_max_Pooling_window_size = int(vectorSize / reverse_top_number) + 1
            
            # generate reverse primers
            seq_T, label_T, seq_V, label_V = process_sequences(sequence_original, 
                                                               vectorSize,
                                                               train_model_number_reverse)
            
            # setup the parameters for the reverse model
            batchSize = reverse_batchSize
            labelSize = 2

            w1 = 12
            wd1 = 21
            h1 = reverse_max_Pooling_window_size
            w4 = 256
            latent_dim_1 = 128
            latent_dim_2 = 64
            latent_dim_3 = 32
            latent_dim_4 = 16
            
            # setup the data for the reverse model
            oneHot_labels_T = oneHot_pytorch(label_T, labelSize)
            oneHot_labels_V = oneHot_pytorch(label_V, labelSize)
            
            seq_T = getBatch_pytorch(seq_T, seq_T.shape[0], vectorSize)
            seq_V = getBatch_pytorch(seq_V, seq_V.shape[0], vectorSize)
            
            train_dataset = TensorDataset(seq_T, oneHot_labels_T)
            val_dataset = TensorDataset(seq_V, oneHot_labels_V)            
            
            train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)
            test_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=False) 
            
            # train the reverse model
            model = ImprovedVAE_CNNModel_V2(vectorSize, labelSize, w1, wd1, h1, w4, latent_dim_1,latent_dim_2, latent_dim_3, latent_dim_4)
            
            trained_model, train_losses, val_losses = train_and_validate(model, train_loader, val_loader, num_epochs, learning_rate, beta, lambda_class)

            precision, recall, f1, diff_frequency, top_diff_positions = evaluate(trained_model, test_loader)
            
            # save the model
            model_path = path + 'model/' + forward_primer + '.pt'
            save_model(trained_model, model_path)
            
            # -------------------------------------------
            # -------------------------------------------
            # -------------------------------------------
            # save the filters
            sequence = sequence_original[train_model_number_reverse: total_number_reverse]
            seq_labels = np.zeros(len(sequence), dtype=int)
            
            pd.DataFrame(sequence).to_csv(f'{path}filter_seq/filter_seq.csv', header=None, index=None)
            data = getBatch_pytorch(sequence, sequence.shape[0], vectorSize)
            
            print(f'\nStart the Filter files...')
            start_time = time.time()
            conv1_output = extract_conv1_output(trained_model, data).cpu().detach().numpy()
            end_time = time.time()
            print(f'Filter files complete!     Time cost: {end_time - start_time}\n')
            print(f'The first Conv1 layer size:      Conv1 = {conv1_output.shape}\n')

            for filterIndex in range(conv1_output.shape[1]):
                print(f'Variant {variant_Name} : Generating the {filterIndex} Filter file')

                Mat = conv1_output[:, filterIndex, :, 0]
                pd.DataFrame(Mat).to_csv(f'{path}filter/filter_{filterIndex}.csv', header=None, index=None)
            
            # -------------------------------------------
            # -------------------------------------------
            # -------------------------------------------
            # reverse primer
            if reverse_method == 'Pooling':
                numberWindows = posPool_pytorch_cuda(path, vectorSize, reverse_max_Pooling_window_size)
            elif reverse_method == 'Top':
                numberWindows = posPool_top_pytorch_cuda(path, vectorSize, reverse_max_Pooling_window_size)
            elif reverse_method == 'Combination':
                numberWindows = posPool_combination_pytorch_cuda(path, vectorSize,
                                                                reverse_max_Pooling_window_size,
                                                                reverse_top_in_window)
                
            
            createFeatVector_pytorch(path, numberWindows, vectorSize, primer_length)
            getFeature(basic_path, path, variant_Name, number = 10)
            p_value = reverse_sameFeature(path)
            
            if p_value == 2:
                pass
            else:
                p_value = reverse_calculateAppearance(basic_path, path, variant_Name)
                if p_value == 2:
                    pass
                else:
                    reverse_commonFeatureAppearance(path, forward_primer)
            
            
        

        
        