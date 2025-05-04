from collections import Counter
import torch
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from functools import reduce
from typing import List, Tuple

from v7_main_model_functions import *
from v7_1_get_data import *

def posPool_pytorch_cuda(path, vectorSize, max_Pooling_window_size):
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    files = glob.glob(os.path.join(path, 'filter', '*.csv'))
    # numberWindows = int(vectorSize / max_Pooling_window_size) + (vectorSize % max_Pooling_window_size > 0)
    numberWindows = int(vectorSize / max_Pooling_window_size)

    for file in files:
        filterIndex = file.split('/')[-1].split('.')[0].split('_')[-1]
        data = pd.read_csv(file, header=None).values
        # Convert data to tensor and move to the specified device
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

        # Max pooling operation
        maxPool, posPool = torch.nn.functional.max_pool1d(data_tensor.unsqueeze(1), 
                                                    max_Pooling_window_size,
                                                    stride=max_Pooling_window_size, return_indices=True)

        # Adjusting indices to reflect the positions in the original data
        posPool = posPool.squeeze().cpu().numpy()

        # Move the max pooled results back to CPU for saving to disk
        maxPool = maxPool.squeeze().cpu().numpy()

        # Save the results
        pd.DataFrame(maxPool).to_csv(os.path.join(path, 'maxPool', 'maxPool_' + str(filterIndex) + '.csv'), header=None,
                                     index=None)
        pd.DataFrame(posPool).to_csv(os.path.join(path, 'posPool', 'posPool_' + str(filterIndex) + '.csv'), header=None,
                                     index=None)

    return numberWindows


def posPool_top_pytorch_cuda(path, vectorSize, max_Pooling_window_size):
    # Set the device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Calculate the number of top values to select based on the vector size and pooling window size
    numberWindows = int(vectorSize / max_Pooling_window_size) + (vectorSize % max_Pooling_window_size > 0)

    # Iterate through the CSV files in the specified directory
    files = glob.glob(os.path.join(path, 'filter', '*.csv'))
    for file in files:
        filterIndex = file.split('/')[-1].split('.')[0].split('_')[-1]
        # Load the data
        data = pd.read_csv(file, header=None).values

        # Convert the data to a PyTorch tensor and move it to the device
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

        # Find the top 'numberWindows' values and their indices in each sequence
        max_values, max_indices = torch.topk(data_tensor, k=numberWindows, dim=1)

        # Move the results back to CPU for saving to disk
        max_values = max_values.cpu().numpy()
        max_indices = max_indices.cpu().numpy()

        # Save the results
        pd.DataFrame(max_values).to_csv(os.path.join(path, 'maxPool', f'maxPool_{filterIndex}.csv'), header=None, index=None)
        pd.DataFrame(max_indices).to_csv(os.path.join(path, 'posPool', f'posPool_{filterIndex}.csv'), header=None, index=None)

    return numberWindows


def posPool_combination_pytorch_cuda(path, vectorSize, max_Pooling_window_size, top_in_window):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    files = glob.glob(os.path.join(path, 'filter', '*.csv'))
    numberWindows = (int(vectorSize / max_Pooling_window_size)) * top_in_window


    for file in files:
        filterIndex = os.path.basename(file).split('.')[0].split('_')[-1]
        data = pd.read_csv(file, header=None).values

        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

        # Initialize the tensors for maxPool and posPool
        maxPool = torch.zeros((data_tensor.size(0), numberWindows), device=device)
        posPool = torch.zeros((data_tensor.size(0), numberWindows), dtype=torch.long, device=device)

        for i, sequence in enumerate(data_tensor):
            temp_values = sequence.unfold(0, max_Pooling_window_size, max_Pooling_window_size)
            window_max_values = torch.zeros((temp_values.size(0) * top_in_window,), device=device)
            window_max_indices = torch.zeros((temp_values.size(0) * top_in_window,), dtype=torch.long, device=device)

            for j, window in enumerate(temp_values):
                top_values, top_indices = torch.topk(window, top_in_window)
                start_idx = j * top_in_window
                window_max_values[start_idx:start_idx + top_in_window] = top_values
                window_max_indices[start_idx:start_idx + top_in_window] = top_indices + j * max_Pooling_window_size

            maxPool[i] = window_max_values
            posPool[i] = window_max_indices

        # Move the results back to CPU for saving to disk
        maxPool = maxPool.cpu().numpy()
        posPool = posPool.cpu().numpy()

        # Save the results
        pd.DataFrame(maxPool).to_csv(os.path.join(path, 'maxPool', f'maxPool_{filterIndex}.csv'), header=None,
                                     index=None)
        pd.DataFrame(posPool).to_csv(os.path.join(path, 'posPool', f'posPool_{filterIndex}.csv'), header=None,
                                     index=None)

    return numberWindows


def createFeatVector_mapping_pytorch(path, numberWindows, vectorSize, numberFilters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_path = os.path.join(path, 'posPool')
    path_seq = os.path.join(path, 'filter_seq')
    files = glob.glob(os.path.join(pos_path, '*.csv'))

    nucleotide_map = {
        'A': 1.00,  # Adenine
        'C': 0.25,  # Cytosine
        'G': 0.75,  # Guanine
        'T': 0.50,  # Thymine
        'N': 0.00,  # Any base
        'U': 0.55,  # Uracil (RNA)
        'R': 0.95,  # A or G
        'Y': 0.35,  # C or T
        'S': 0.60,  # G or C
        'W': 0.80,  # A or T
        'K': 0.65,  # G or T
        'M': 0.70,  # A or C
        'B': 0.40,  # C or G or T
        'D': 0.85,  # A or G or T
        'H': 0.45,  # A or C or T
        'V': 0.90,  # A or C or G
        '-': -0.10,  # Gap
        '.': -0.20  # Gap
    }
    inv_map = {v: k for k, v in nucleotide_map.items()}

    if numberFilters % 2 == 0:
        padding = int(numberFilters / 2)
    else:
        padding = int((numberFilters - 1) / 2)

    for file in files:
        filterIndex = os.path.basename(file).split('.')[0].split('_')[-1]
        print(f'Processing... Loop 1 -- Index {filterIndex}')

        posMatrix = pd.read_csv(file, header=None).values
        posMatrix_tensor = torch.tensor(posMatrix, dtype=torch.long).to(device)
        sequence_data = pd.read_csv(os.path.join(path_seq, 'filter_seq.csv'), header=None).values.ravel()

        dataDNAFeatures = []
        for sequence in sequence_data:
            sample = [nucleotide_map[nuc] for nuc in sequence]
            sample_tensor = torch.tensor(sample, dtype=torch.float32).to(device)
            
            padding_size = vectorSize - len(sample_tensor)
            if padding_size <= 0:
                pass
            else:
                sample_tensor = torch.nn.functional.pad(sample_tensor, (0, padding_size), mode='constant', value=0.00)

            temp_features = torch.zeros((posMatrix_tensor.shape[0], numberWindows * numberFilters), device=device)

            for i in range(posMatrix_tensor.shape[0]):
                for j in range(posMatrix_tensor.shape[1]):
                    start_idx = max(posMatrix_tensor[i, j] - padding, 0)
                    end_idx = min(posMatrix_tensor[i, j] + padding + 1, vectorSize)
                    temp_features[i, j * numberFilters:(j * numberFilters + end_idx - start_idx)] = sample_tensor[
                                                                                                    start_idx:end_idx]

            # Convert numeric features back to nucleotide sequences
            temp_features_cpu = temp_features.cpu().numpy()
            for row in temp_features_cpu:
                feature_str = ''.join([inv_map[val] for val in row])
                dataDNAFeatures.append(feature_str)

        # This part simplifies and deduplicates features, preserving only unique feature strings
        unique_features = list(set(dataDNAFeatures))

        # Write unique features to a CSV file
        pd.DataFrame(unique_features).to_csv(os.path.join(path, 'featsVector', f'featsVector_{filterIndex}.csv'), header=None, index=None)


def createFeatVector_pytorch(path, numberWindows, vectorSize, numberFilters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_path = os.path.join(path, 'posPool')
    path_seq = os.path.join(path, 'filter_seq')
    files = glob.glob(os.path.join(pos_path, '*.csv'))

    if numberFilters % 2 == 0:
        padding = int(numberFilters / 2)
    else:
        padding = int((numberFilters - 1) / 2)

    for file in files:
        filterIndex = os.path.basename(file).split('.')[0].split('_')[-1]
        print(f'Processing... Loop 1 -- Index {filterIndex}')

        posMatrix = pd.read_csv(file, header=None).values
        posMatrix_tensor = torch.tensor(posMatrix, dtype=torch.long).to(device)
        sequence_data = pd.read_csv(os.path.join(path_seq, 'filter_seq.csv'), header=None).values.ravel()

        dataDNAFeatures = []
        for sequence in sequence_data:
            # Convert sequence to list of characters and pad if necessary
            sample = list(sequence)
            if len(sample) < vectorSize:
                sample.extend(['N'] * (vectorSize - len(sample)))
            elif len(sample) > vectorSize:
                sample = sample[:vectorSize]

            for i in range(posMatrix_tensor.shape[0]):
                for j in range(posMatrix_tensor.shape[1]):
                    start_idx = max(posMatrix_tensor[i, j].item() - padding, 0)
                    end_idx = min(posMatrix_tensor[i, j].item() + padding + 1, vectorSize)
                    temp = ''.join(sample[start_idx:end_idx])
                    dataDNAFeatures.append(temp)

        # Simplify and deduplicate features, preserving only unique feature strings
        unique_features = list(set(dataDNAFeatures))

        # Write unique features to a CSV file
        pd.DataFrame(unique_features).to_csv(os.path.join(path, 'featsVector', f'featsVector_{filterIndex}.csv'), header=None, index=None)


def count_set_elements_in_string(s, element_set):
    counts = Counter()
    for element in element_set:
        # 如果元素是单个字符，直接计数
        if len(element) == 1:
            counts[element] = s.count(element)
        else:
            # 对于多字符元素，使用滑动窗口计数
            count = 0
            for i in range(len(s) - len(element) + 1):
                if s[i:i+len(element)] == element:
                    count += 1
            counts[element] = count
    return counts

def getFeature(basic_path, path, variant_name, number):
    files = glob.glob(os.path.join(path, 'featsVector', '*.csv'))
    seq_file = os.path.join(basic_path, 'seq_and_seqName', f'seq_GISAID_{variant_name}.csv')

    sequences = pd.read_csv(seq_file, header=None).values.ravel()
    np.random.shuffle(sequences)
    sequences = sequences[:number]

    for file in tqdm(files, desc="Processing files"):
        filter_index = os.path.basename(file).split('.')[0].split('_')[-1]
        print(f'Processing file: {os.path.basename(file)}')

        vector = set(pd.read_csv(file, header=None).values.ravel())
        print(f'Sequence Size: {len(sequences)}, featVector Size: {len(vector)}')

        feature_counter = Counter()

        for seq in tqdm(sequences, desc="Analyzing sequences"):
            # Count occurrences of each feature in the sequence
            seq_features = count_set_elements_in_string(seq, vector)
            # Only keep features that occur exactly once
            unique_features = {feat for feat, count in seq_features.items() if count == 1}
            feature_counter.update(unique_features)

        # Keep features that appear in at least one sequence
        final_features = [feat for feat, count in feature_counter.items() if count > 0]

        if final_features:
            output_file = os.path.join(path, 'feature', f'features_{filter_index}.csv')
            pd.DataFrame(final_features).to_csv(output_file, header=None, index=None)
            print(f'Saved {len(final_features)} features to {output_file}')
        else:
            print(f'No features found for filter index {filter_index}')

    print('Feature extraction complete.')


def sameFeature(path):
    feature_files = glob.glob(os.path.join(path, 'feature', '*'))

    # Use a set to store unique features and count occurrences
    feature_counts = Counter()

    print("Reading and processing feature files...")
    for file in tqdm(feature_files, desc="Processing files"):
        features = pd.read_csv(file, header=None).values.ravel()
        feature_counts.update(features)

    print(f"Total unique features: {len(feature_counts)}")

    # Filter features
    repeat_features = []
    cg_filtered_features = []

    print("Filtering features...")
    for feature, count in tqdm(feature_counts.items(), desc="Filtering features"):
        if count > 1:
            repeat_features.append(feature)
        
        cg_count = feature.count('C') + feature.count('G')
        cg_ratio = cg_count / len(feature)
        if 'C' in feature and 'G' in feature and 0.3 < cg_ratio < 0.7:
            cg_filtered_features.append(feature)

    print(f"Repeat features: {len(repeat_features)}")
    print(f"CG filtered features: {len(cg_filtered_features)}")

    # Save the filtered lists
    repeat_file = os.path.join(path, 'Repeat_feature_List.csv')
    cg_file = os.path.join(path, 'CG_filtered_feature_List.csv')

    pd.DataFrame(repeat_features).to_csv(repeat_file, header=None, index=None)
    pd.DataFrame(cg_filtered_features).to_csv(cg_file, header=None, index=None)

    print(f"Saved repeat features to: {repeat_file}")
    print(f"Saved CG filtered features to: {cg_file}")


def get_appearance(feature_file, seq_file, high_type=1, accuracy=0.95, sample_size=5000):
    features = pd.read_csv(feature_file, header=None).values.ravel()
    seq = pd.read_csv(seq_file, header=None).values.ravel()
    
    # Sample sequences
    np.random.shuffle(seq)
    seq = seq[:sample_size]
    total_num = len(seq)

    print(f"Processing {len(features)} features across {total_num} sequences...")

    feature_dic = {}

    for feature in tqdm(features, desc="Processing features"):
        # Count the occurrence of each feature in all sequences
        count_feature = np.array([s.count(feature) for s in seq])
        available_count = np.sum(count_feature == 1)

        appearance_rate = available_count / total_num
        if (high_type == 1 and appearance_rate >= accuracy) or \
                (high_type != 1 and appearance_rate <= 1 - accuracy):
            feature_dic[feature] = appearance_rate


    feature_df = pd.DataFrame.from_dict(feature_dic, orient='index', columns=['Appearance Rate'])
    
    print(f"Found {len(feature_dic)} features meeting the criteria.")
    return feature_df


def calculateAppearance(basic_path, path, variant_Name):
    files = glob.glob(os.path.join(basic_path, 'Variant_virus', '*'))
    order = [3, 1, 0, 2, 4]  # original order = ['alpha', 'beta', 'gamma', 'delta', 'omicron']
    feature_file = os.path.join(path, 'CG_filtered_feature_List.csv')
    temp_feature_file = os.path.join(path, 'result', 'temp_feature.csv')

    def get_params(save_fileName):
        if variant_Name == 'omicron':
            if save_fileName == 'omicron':
                return 1, 0.80
            elif save_fileName == 'delta':
                return 1, -1
            else:
                return 0, 0.90
        else:
            if save_fileName == 'omicron':
                return 1, -1
            elif save_fileName == variant_Name:
                return 1, 0.95
            else:
                return 0, 0.95

    for i in tqdm(order, desc="Processing variants"):
        file = os.path.basename(files[i])
        save_fileName = file.split('.')[0].replace('GISAID_', '')
        seq_file = os.path.join(basic_path, 'seq_and_seqName', f'seq_GISAID_{save_fileName}.csv')

        print(f'Get the appearance in -- {save_fileName} -- virus')
        
        high_type, accuracy = get_params(save_fileName)
        
        current_feature_file = temp_feature_file if os.path.exists(temp_feature_file) else feature_file
        
        featureDF = get_appearance(current_feature_file, seq_file, high_type=high_type, accuracy=accuracy)
        output_file = os.path.join(path, 'Seq_appearance', f'feature_{save_fileName}.csv')
        featureDF.to_csv(output_file)

        # Update temp_feature file
        temp_feature = featureDF.index.values
        pd.DataFrame(temp_feature).to_csv(temp_feature_file, header=None, index=None)

    print("Appearance calculation completed for all variants.")


def commonFeatureAppearance(path):
    files = glob.glob(os.path.join(path, 'Seq_appearance', '*'))

    # Read all files at once and store in a dictionary
    data_dict = {}
    for file in files:
        file_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, header=None, names=['feature', 'appearance'])
        df = df.set_index('feature')['appearance']
        data_dict[file_name] = df

    # Find common features efficiently
    common_features = list(reduce(set.intersection, (set(df.index) for df in data_dict.values())))

    # Create the appearance DataFrame more efficiently
    appearance_df = pd.DataFrame({name: df.reindex(common_features).fillna(0)
                                  for name, df in data_dict.items()})

    # Sorting the DataFrame by specific columns if all required columns are present
    required_columns = ['feature_alpha', 'feature_beta', 'feature_gamma', 'feature_delta', 'feature_omicron']
    if all(col in appearance_df.columns for col in required_columns):
        appearance_df = appearance_df[required_columns]

    # Optionally sort by 'feature_omicron' if present
    if 'feature_omicron' in appearance_df.columns:
        appearance_df = appearance_df.sort_values(by='feature_omicron', ascending=False)

    # Save the result without generating the Appearance Rate row
    index_list = list(appearance_df.index)
    del index_list[0]

    appearance_df = appearance_df.reset_index(drop=True)
    appearance_df = appearance_df.drop([0])

    if len(appearance_df) == len(index_list):
        appearance_df.index = index_list
    else:
        print("Warning: The length of my_list does not match the number of rows in the DataFrame")

    appearance_df.to_csv(os.path.join(path, 'Appearance_DataFrame.csv'))
    

# ----------------------------------------------------------- #    
# ----------------- Reverse Model Functions ----------------- #
# ----------------------------------------------------------- #


def get_forward_primers(path):
    print('------------------ Processing get_data ------------------')
    
    files = glob.glob(os.path.join(path, 'forward_primer', '*'))
    
    forward_primer_set = set()
    total_count = 0
    
    print('Starting to collect forward primers...')
    for file in files:
        primers = pd.read_csv(file)['Unnamed: 0'].values.ravel()
        total_count += len(primers)
        forward_primer_set.update(primers)
    
    print(f'Saving {len(forward_primer_set)} unique forward primers...')
    forward_primer_list = list(forward_primer_set)
    pd.DataFrame(forward_primer_list).to_csv(os.path.join(path, 'forward_primer.csv'), header=False, index=False)
    
    print(f'Finished!     ({len(forward_primer_set)}/{total_count})\n')
    

def CG_content_check(path, min_CG = 0.35, max_CG = 0.65):
    print('Checking the CG content of the forward primers obtained....')

    features = pd.read_csv(path + 'forward_primer.csv', header=None).values.ravel()

    def is_valid_cg_content(feature: str) -> bool:
        cg_count = sum(feature.count(base) for base in 'CG')
        return len(feature) * min_CG <= cg_count <= len(feature) * max_CG

    new_features = list(set(feature for feature in features if 'C' in feature and 'G' in feature and is_valid_cg_content(feature)))

    print(f'Finished!     ({len(new_features)}/{len(features)})        min={min_CG}  max={max_CG}')
    
    pd.DataFrame(new_features).to_csv(path + 'forward_primer_CG_check.csv', header=None, index=None)
    
    
def get_after_primer_data(path, forward_primer, sequence):

    print('\nNow processing with forward primer:  {}\n'.format(forward_primer))
    in_count, out_count = 0, 0
    second_half_list = []

    np.random.shuffle(sequence)
    sequence = sequence[:6000]
    for i in range(len(sequence)):
        if i % 5000 == 0:
            print('No. {} sequence... with {} in the sequence     /     with {} out of the sequence'.
                  format(i, in_count, out_count))
        if forward_primer in sequence[i]:
            in_count += 1
            second_half = sequence[i].split(forward_primer)[1]
            second_half_list.append(second_half)
        else:
            out_count += 1
            pass
    print('\n{} : {} sequence\n'.format(forward_primer, len(second_half_list)))
    # pd.DataFrame(second_half_list).to_csv(path + 'second_data/' + str(forward_primer) + '.csv', header=None,
    #                                         index=None)
    return pd.DataFrame(second_half_list).values.ravel()


def get_after_primer_data(path, forward_primer, sequence, sample_size=5000):
    print(f'\nNow processing with forward primer: {forward_primer}\n')
    
    # Shuffle the sequence
    np.random.shuffle(sequence)
    
    second_half_list = []
    
    # Use tqdm for progress bar
    pbar = tqdm(total=sample_size, desc="Processing sequences")
    
    for seq in sequence:
        if forward_primer in seq:
            second_half = seq.split(forward_primer)[1]
            second_half_list.append(second_half)
            pbar.update(1)
            
            if len(second_half_list) >= sample_size:
                break
    
    pbar.close()
    
    print(f'\n{forward_primer}: {len(second_half_list)} sequences\n')
    
    # Uncomment the following line if you want to save the results to a CSV file
    # pd.DataFrame(second_half_list).to_csv(path + 'second_data/' + str(forward_primer) + '.csv', header=None, index=None)
    
    return pd.DataFrame(second_half_list).values.ravel()


def exist_primer_check(path, forward_primer, sequence):
    primers = pd.read_csv(path + 'forward_primer.csv', header=None).values.ravel()
    print(f'Starting the exist primer check... {len(primers)} primers in total\n')
    
    print(f'Processing forward primer: {forward_primer}')

    sequence_df = pd.DataFrame(sequence)
    min_num = int(len(sequence) * 0.99)
    
    reverse_primers = {}
    
    for primer in tqdm(primers, desc="Checking primers"):
        if forward_primer != primer:
            count_feature = sequence_df[0].apply(lambda x: x.count(primer))
            available_count = (count_feature == 1).sum()
            if available_count >= min_num:
                reverse_primers[primer] = available_count / len(sequence)
    
    if reverse_primers:
        print('Matching reverse primers found')
        reverse_primers_df = pd.DataFrame.from_dict(reverse_primers, orient='index', columns=['Frequency'])
        reverse_primers_df.to_csv(path + 'exist_primer_check/' + forward_primer + '.csv')
    else:
        print('No matching reverse primers found')
    
    print('\nExist primer check completed!\n')



def generate_random_sequence(sequence, number, vectorSize):

     # Calculate base percentages
    base_counts = Counter()
    for seq in sequence:
        base_counts.update(seq)
    
    total_bases = sum(base_counts.values())
    base_percentages = {base: count / total_bases for base, count in base_counts.items()}
    
    # Print average base percentages
    for base in 'ATCG':
        print(f'Average "{base}" in the sequence      {base} --    {base_percentages.get(base, 0):.4f}')
    print()
    
    # Generate base list for random sequences
    ATCG_random_list = []
    for base, percentage in base_percentages.items():
        ATCG_random_list.extend([base] * round(vectorSize * percentage))
    
    # Adjust list length if necessary
    if len(ATCG_random_list) != vectorSize:
        if len(ATCG_random_list) > vectorSize:
            ATCG_random_list = ATCG_random_list[:vectorSize]
        else:
            ATCG_random_list.extend(['N'] * (vectorSize - len(ATCG_random_list)))
    
    # Generate random sequences
    def generate_random_data(n):
        random_data = []
        for _ in range(n):
            np.random.shuffle(ATCG_random_list)  # Shuffle before each sequence generation
            random_data.append(''.join(ATCG_random_list))
        return random_data
    
    random_data_T = generate_random_data(number)
    random_data_V = generate_random_data(number)
    
    return random_data_T, random_data_V


def process_sequences(sequence_original, vectorSize, train_model_number_reverse):
        
    # Slice the original sequence
    sequence = sequence_original[:train_model_number_reverse]
    number = len(sequence)
    
    # Generate random sequences
    random_sequence_T, random_sequence_V = generate_random_sequence(sequence, number, vectorSize)
    
    # Create labels
    labels = np.zeros(number, dtype=int)
    labels_random = np.ones(number, dtype=int)
    
    # Combine sequences and labels for training and validation
    seq_T = np.concatenate([sequence, random_sequence_T])
    label_T = np.concatenate([labels, labels_random])
    
    seq_V = np.concatenate([sequence, random_sequence_V])
    label_V = np.concatenate([labels, labels_random])
    
    # Shuffle training data
    rand = np.random.randint(100000)
    np.random.seed(rand)
    shuffle_indices = np.random.permutation(len(seq_T))
    seq_T = seq_T[shuffle_indices]
    label_T = label_T[shuffle_indices]
    
    # Shuffle validation data
    rand = np.random.randint(100000)
    np.random.seed(rand)
    shuffle_indices = np.random.permutation(len(seq_V))
    seq_V = seq_V[shuffle_indices]
    label_V = label_V[shuffle_indices]
    
    return seq_T, label_T, seq_V, label_V

    
def reverse_sameFeature(path):
    feature_files = glob.glob(os.path.join(path, 'feature', '*'))

    # Use a set to store unique features and count occurrences
    feature_counts = Counter()

    print("Reading and processing feature files...")
    for file in tqdm(feature_files, desc="Processing files"):
        features = pd.read_csv(file, header=None).values.ravel()
        feature_counts.update(features)

    print(f"Total unique features: {len(feature_counts)}")

    # Filter features
    repeat_features = []
    cg_filtered_features = []

    print("Filtering features...")
    for feature, count in tqdm(feature_counts.items(), desc="Filtering features"):
        if count > 1:
            repeat_features.append(feature)
        
        cg_count = feature.count('C') + feature.count('G')
        cg_ratio = cg_count / len(feature)
        if 'C' in feature and 'G' in feature and 0.3 < cg_ratio < 0.7:
            cg_filtered_features.append(feature)

    
    if len(cg_filtered_features) == 0:
        p_value = 2
    else:
        p_value = 1
        print(f"Repeat features: {len(repeat_features)}")
        print(f"CG filtered features: {len(cg_filtered_features)}")

        # Save the filtered lists
        repeat_file = os.path.join(path, 'Repeat_feature_List.csv')
        cg_file = os.path.join(path, 'CG_filtered_feature_List.csv')

        pd.DataFrame(repeat_features).to_csv(repeat_file, header=None, index=None)
        pd.DataFrame(cg_filtered_features).to_csv(cg_file, header=None, index=None)

        print(f"Saved repeat features to: {repeat_file}")
        print(f"Saved CG filtered features to: {cg_file}")
        
    return p_value


def reverse_calculateAppearance(basic_path, path, variant_Name):
    files = glob.glob(os.path.join(basic_path, 'Variant_virus', '*'))
    order = [3, 1, 0, 2, 4]  # original order = ['alpha', 'beta', 'gamma', 'delta', 'omicron']
    feature_file = os.path.join(path, 'CG_filtered_feature_List.csv')
    temp_feature_file = os.path.join(path, 'result', 'temp_feature.csv')
    
    p_value = 0

    def get_params(save_fileName):
        if variant_Name == 'omicron':
            if save_fileName == 'omicron':
                return 1, 0.80
            elif save_fileName == 'delta':
                return 1, -1
            else:
                return 0, 0.90
        else:
            if save_fileName == 'omicron':
                return 1, -1
            elif save_fileName == variant_Name:
                return 1, 0.95
            else:
                return 0, 0.95

    for i in tqdm(order, desc="Processing variants"):
        file = os.path.basename(files[i])
        save_fileName = file.split('.')[0].replace('GISAID_', '')
        seq_file = os.path.join(basic_path, 'seq_and_seqName', f'seq_GISAID_{save_fileName}.csv')

        print(f'Get the appearance in -- {save_fileName} -- virus')
        
        high_type, accuracy = get_params(save_fileName)
        
        if p_value == 0:
            current_feature_file = feature_file
            featureDF = get_appearance(current_feature_file, seq_file, high_type=high_type, accuracy=accuracy)
            output_file = os.path.join(path, 'Seq_appearance', f'feature_{save_fileName}.csv')
            featureDF.to_csv(output_file)
            
            # Update temp_feature file
            temp_feature = featureDF.index.values
            pd.DataFrame(temp_feature).to_csv(temp_feature_file, header=None, index=None)
            
            if len(temp_feature) == 0:
                p_value = 2
            else:
                p_value = 1
        
        elif p_value == 1:
            current_feature_file = temp_feature_file if os.path.exists(temp_feature_file) else feature_file
            featureDF = get_appearance(current_feature_file, seq_file, high_type=high_type, accuracy=accuracy)
            output_file = os.path.join(path, 'Seq_appearance', f'feature_{save_fileName}.csv')
            featureDF.to_csv(output_file)

            # Update temp_feature file
            temp_feature = featureDF.index.values
            pd.DataFrame(temp_feature).to_csv(temp_feature_file, header=None, index=None)
            
            if len(temp_feature) == 0:
                p_value = 2
        
        elif p_value == 2:
            print('No need to process the rest of the variants')
            pass

    print("Appearance calculation completed for all variants.")
    return p_value


def reverse_commonFeatureAppearance(path, forward_primer):
    files = glob.glob(os.path.join(path, 'Seq_appearance', '*'))

    # Read all files at once and store in a dictionary
    data_dict = {}
    for file in files:
        file_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, header=None, names=['feature', 'appearance'])
        df = df.set_index('feature')['appearance']
        data_dict[file_name] = df

    # Find common features efficiently
    common_features = list(reduce(set.intersection, (set(df.index) for df in data_dict.values())))

    # Create the appearance DataFrame more efficiently
    appearance_df = pd.DataFrame({name: df.reindex(common_features).fillna(0)
                                  for name, df in data_dict.items()})

    # Sorting the DataFrame by specific columns if all required columns are present
    required_columns = ['feature_alpha', 'feature_beta', 'feature_gamma', 'feature_delta', 'feature_omicron']
    if all(col in appearance_df.columns for col in required_columns):
        appearance_df = appearance_df[required_columns]

    # Optionally sort by 'feature_omicron' if present
    if 'feature_omicron' in appearance_df.columns:
        appearance_df = appearance_df.sort_values(by='feature_omicron', ascending=False)

    # Save the result without generating the Appearance Rate row
    index_list = list(appearance_df.index)
    del index_list[0]

    appearance_df = appearance_df.reset_index(drop=True)
    appearance_df = appearance_df.drop([0])

    if len(appearance_df) == len(index_list):
        appearance_df.index = index_list
    else:
        print("Warning: The length of my_list does not match the number of rows in the DataFrame")

    appearance_df.to_csv(path + 'result/' + forward_primer + '_Appearance_DataFrame.csv')
    