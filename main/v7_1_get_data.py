import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import StratifiedKFold
from v7_main_functions import *


def get_data(path):
    print('------------------ Processing get_data ------------------')
    variant_list = ['alpha', 'beta', 'gamma', 'delta', 'omicron']

    for Name in variant_list:
        print('\nNow running the --- {} --- variant\n'.format(Name))
        files_pattern = path + 'Variant_virus/GISAID_' + Name + '/*'
        files = glob.glob(files_pattern)
        seqName_list, seq_list = [], []
        file_count, data_count = 0, 0

        for file in files:
            file_count += 1
            for seqName, seq in readFASTA_iter(file):  # Assuming this is an optimized, external function
                data_count += 1
                seqName_list.append(seqName)
                seq_list.append(seq)
                if data_count % 1000 == 0:
                    print('{} ... No.{} file with {} sequences'.format(Name, file_count, data_count))

        # Log final counts before saving
        print('\n{} : {} sequences\n'.format(Name, len(seqName_list)))

        # Saving the data in one go outside the inner loop
        seqName_df = pd.DataFrame(seqName_list)
        seq_df = pd.DataFrame(seq_list)

        seqName_df.to_csv(path + 'Seq_and_SeqName/seqName_GISAID_' + Name + '.csv', header=None, index=None)
        seq_df.to_csv(path + 'Seq_and_SeqName/seq_GISAID_' + Name + '.csv', header=None, index=None)

        print(f'\nData for {Name} variant saved.\n')


def mixed_data(basic_path, delta_num, other_percent=1):
    variant_list = ['alpha', 'beta', 'gamma', 'delta', 'omicron']
    vectorSize = 0
    mix_seqName, mix_sequence, mix_label = [], [], []

    # Generate random seed outside the loop for consistency
    rand = np.random.randint(100000)

    for i, Name in enumerate(variant_list):
        # Adjust the count for non-delta variants
        count = delta_num if Name == 'delta' else int(delta_num * other_percent)
        label = i

        seqName_file = f'{basic_path}seq_and_seqName/seqName_GISAID_{Name}.csv'
        sequence_file = f'{basic_path}seq_and_seqName/seq_GISAID_{Name}.csv'

        seqName = pd.read_csv(seqName_file, header=None).values.ravel()
        sequence = pd.read_csv(sequence_file, header=None).values.ravel()

        # Update the vectorSize based on sequence lengths
        vectorSize = max(vectorSize, max(len(s) for s in sequence))

        # Shuffle once with the same seed for consistency
        np.random.seed(rand)
        indices = np.random.permutation(len(seqName))[:count]

        mix_seqName.extend(seqName[indices])
        mix_sequence.extend(sequence[indices])
        mix_label.extend([label] * count)

    # Saving all at once to minimize disk I/O
    pd.DataFrame(mix_seqName).to_csv(f'{basic_path}mix_seqName.csv', header=None, index=None)
    pd.DataFrame(mix_sequence).to_csv(f'{basic_path}mix_sequence.csv', header=None, index=None)
    pd.DataFrame(mix_label).to_csv(f'{basic_path}mix_label.csv', header=None, index=None)

    # save vectorSize
    pd.DataFrame([vectorSize]).to_csv(f'{basic_path}vectorSize.csv', header=None, index=None)

    return vectorSize


def train_valid_data(basic_path, n_splits=2):
    # Load mixed data
    mix_seqName = pd.read_csv(f'{basic_path}mix_seqName.csv', header=None).values.flatten()
    mix_sequence = pd.read_csv(f'{basic_path}mix_sequence.csv', header=None).values.flatten()
    mix_label = pd.read_csv(f'{basic_path}mix_label.csv', header=None).values.flatten()

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits)

    # Split data and save each split
    for i, (train_index, test_index) in enumerate(skf.split(mix_sequence, mix_label)):
        # Splitting the data
        mix_sequence_train, mix_sequence_test = mix_sequence[train_index], mix_sequence[test_index]
        mix_seqName_train, mix_seqName_test = mix_seqName[train_index], mix_seqName[test_index]
        mix_label_train, mix_label_test = mix_label[train_index], mix_label[test_index]

        # Saving the splits - appending an index if multiple splits
        suffix = f"_{i}" if n_splits > 2 else ""
        pd.DataFrame(mix_sequence_train).to_csv(f'{basic_path}index/train_sequence{suffix}.csv', header=None, index=None)
        pd.DataFrame(mix_sequence_test).to_csv(f'{basic_path}index/valid_sequence{suffix}.csv', header=None, index=None)

        pd.DataFrame(mix_seqName_train).to_csv(f'{basic_path}index/train_seqName{suffix}.csv', header=None, index=None)
        pd.DataFrame(mix_seqName_test).to_csv(f'{basic_path}index/valid_seqName{suffix}.csv', header=None, index=None)

        pd.DataFrame(mix_label_train).to_csv(f'{basic_path}index/train_label{suffix}.csv', header=None, index=None)
        pd.DataFrame(mix_label_test).to_csv(f'{basic_path}index/valid_label{suffix}.csv', header=None, index=None)


if __name__ == '__main__':
    basic_path = './'  # Assuming this is the default path
    sys_path(basic_path)

    get_data(basic_path) # Run only once