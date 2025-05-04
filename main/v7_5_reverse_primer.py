import numpy as np
import pandas as pd
import glob
from collections import Counter
import os
import shutil
import primer3
from tqdm import tqdm


def get_after_primer_data(path, forward_primer, sequence, number = 5000):
    print(f'\nProcessing with forward primer: {forward_primer}\n')

    # Randomly select sequences
    np.random.shuffle(sequence)
    sequence = sequence[:number]

    # Use list comprehension and string split method
    second_half_list = [seq.split(forward_primer)[1] for seq in tqdm(sequence) if forward_primer in seq]

    in_count = len(second_half_list)
    out_count = len(sequence) - in_count

    print(f'\nSequences containing primer {forward_primer}: {in_count}')
    print(f'Sequences not containing primer: {out_count}')
    print(f'\n{forward_primer} : {in_count} sequences\n')

    return pd.DataFrame(second_half_list).values.ravel()


def get_available_primers(path):
    print('Getting available primers with 40%-60% CG content...')

    # Remove temp feature file if exists
    temp_feature_path = os.path.join(path, 'result', 'temp_feature.csv')
    if os.path.exists(temp_feature_path):
        os.remove(temp_feature_path)

    # Read features
    features = set(pd.read_csv(os.path.join(path, 'forward_primer_CG_check.csv'), header=None).values.ravel())

    # Process result files
    result_files = glob.glob(os.path.join(path, 'result', '*'))
    for file in tqdm(result_files, desc="Processing result files"):
        forward_primer = os.path.basename(file).split('_')[0]
        if forward_primer in features:
            new_path = os.path.join(path, 'CG_Check', 'new_primers', f'{forward_primer}.csv')
            shutil.copyfile(file, new_path)

    # Process exist primer check files
    exist_files = glob.glob(os.path.join(path, 'exist_primer_check', '*'))
    for file in tqdm(exist_files, desc="Processing existing primer files"):
        forward_primer = os.path.basename(file).split('.')[0]
        if forward_primer in features:
            new_path = os.path.join(path, 'CG_Check', 'exist_primers', f'{forward_primer}.csv')
            shutil.copyfile(file, new_path)

    print('Finished!\n')


def calculate_CG_content_and_melting_temperature(primer):
    # Convert primer to uppercase to ensure consistency
    primer = primer.upper()

    # Check if primer contains only valid nucleotides
    valid_nucleotides = set('ATCG')
    if not set(primer).issubset(valid_nucleotides):
        return -1, -1

    # Count nucleotides
    nucleotide_counts = Counter(primer)
    total_count = len(primer)

    # Calculate CG content
    cg_count = nucleotide_counts['C'] + nucleotide_counts['G']
    cg_content = cg_count / total_count

    # Calculate melting temperature
    tm = 64.9 + 41 * (cg_count - 16.4) / total_count

    return cg_content, tm


def process_primers(path, primer_type, sequence_full, feature_CG_Check):
    print(f'\nProcessing {primer_type} primers:')
    primer_design = []
    files = glob.glob(os.path.join(path, 'CG_Check', f'{primer_type}_primers', '*'))

    for file in tqdm(files, desc=f"Processing {primer_type} primers"):
        forward_primer = os.path.basename(file).split('.')[0]
        sequence = get_after_primer_data(path, forward_primer, sequence_full)
        reverse_primers = pd.read_csv(file)['Unnamed: 0'].values

        for reverse_primer in reverse_primers:
            if primer_type == 'new' or reverse_primer in feature_CG_Check:
                lengths = [len(seq.split(reverse_primer)[0]) + len(forward_primer) + len(reverse_primer)
                           for seq in sequence if reverse_primer in seq]

                if lengths:
                    f_CG, f_Tm = calculate_CG_content_and_melting_temperature(forward_primer)
                    r_CG, r_Tm = calculate_CG_content_and_melting_temperature(reverse_primer)

                    primer_design.append([
                        forward_primer, reverse_primer, f_CG, r_CG, f_Tm, r_Tm, abs(f_Tm - r_Tm),
                        np.mean(lengths), np.max(lengths), np.min(lengths)
                    ])

    pd.DataFrame(primer_design, columns=[
        'Forward Primer', 'Reverse Primer', 'Forward CG', 'Reverse CG', 'Forward Tm', 'Reverse Tm', 'Tm Difference',
        'Mean Length', 'Max Length', 'Min Length'
    ]).to_csv(os.path.join(path, 'amplicon_length', f'{primer_type}_primers.csv'), index=False)


def length_of_amplicon(path, sequence_full):
    print('Get the length of amplicon between the primers...\n')
    feature_CG_Check = pd.read_csv(os.path.join(path, 'forward_primer_CG_check.csv'), header=None).values.ravel()

    for primer_type in ['exist', 'new']:
        process_primers(path, primer_type, sequence_full, feature_CG_Check)

    print("\nFinished processing all primers.")


def reshape_file(path):
    columns = [
        'Forward primer', 'Reverse primer', 'Forward CG content', 'Reverse CG content',
        'Forward Melting Temperature (Tm)', 'Reverse Melting Temperature (Tm)', 'Tm difference',
        'amplicon_avg', 'amplicon_max', 'amplicon_min'
    ]

    for primer_type in ['new', 'exist']:
        file_path = os.path.join(path, 'amplicon_length', f'{primer_type}_primers.csv')
        df = pd.read_csv(file_path, header=None)
        df = pd.DataFrame(df.values.reshape(-1, len(columns)), columns=columns)
        df.to_csv(file_path, index=False)

    print("Files reshaped successfully.")


def reverse_complement(sequence):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement.get(base, base) for base in reversed(sequence))


def calculate_tm(sequence):
    if len(set(sequence)) == 4:
        counter = Counter(sequence)
        return 64.9 + 41 * (counter['G'] + counter['C'] - 16.4) / len(sequence)
    return -1


def process_primers_2(df):
    df['Reverse primer'] = df['Reverse primer'].apply(reverse_complement)
    df['Reverse Melting Temperature (Tm)'] = df['Reverse primer'].apply(calculate_tm)

    df['Forward Melting Temperature (Tm)'] = pd.to_numeric(df['Forward Melting Temperature (Tm)'], errors='coerce')
    df['Reverse Melting Temperature (Tm)'] = pd.to_numeric(df['Reverse Melting Temperature (Tm)'], errors='coerce')
    df['Tm difference'] = np.abs(df['Forward Melting Temperature (Tm)'] - df['Reverse Melting Temperature (Tm)'])
    return df


def reverse(path):
    for primer_type in ['exist', 'new']:
        print(f'\nProcessing {primer_type} primers:')
        file_path = os.path.join(path, 'amplicon_length', f'{primer_type}_primers.csv')
        df = pd.read_csv(file_path)
        df = df.drop(0)
        df = process_primers_2(df)
        df.to_csv(os.path.join(path, 'amplicon_length', f'r_{primer_type}_primers.csv'), index=False)
        print(f'Finished processing {primer_type} primers!')

    print("\nAll primers processed successfully.")

def primer_design_rules(path):
    for primer_type in ['exist', 'new']:
        print(f'\nProcessing {primer_type} primers:')
        input_file = os.path.join(path, 'amplicon_length', f'r_{primer_type}_primers.csv')
        output_file = os.path.join(path, 'amplicon_length', f'r_{primer_type}_primers_deltaG.csv')

        primers = pd.read_csv(input_file)
        valid_primers = []

        for _, row in tqdm(primers.iterrows(), total=len(primers), desc="Processing primers"):
            forward_primer = row['Forward primer']
            reverse_primer = row['Reverse primer']

            # Check if primers end with C or G
            if ('C' not in forward_primer[-3:] and 'G' not in forward_primer[-3:]) or \
                    ('C' not in reverse_primer[-3:] and 'G' not in reverse_primer[-3:]):
                continue

            # Check homodimer and heterodimer
            if primer3.calc_homodimer(forward_primer).dg <= -9000 or \
                    primer3.calc_homodimer(reverse_primer).dg <= -9000 or \
                    primer3.calc_heterodimer(forward_primer, reverse_primer).dg <= -9000:
                continue

            # Calculate and check melting temperatures
            f_tm = primer3.calc_tm(forward_primer)
            r_tm = primer3.calc_tm(reverse_primer)

            if abs(f_tm - r_tm) > 5 or not (50 <= f_tm <= 60) or not (50 <= r_tm <= 60):
                continue

            # If all checks pass, update the row and add to valid primers
            row['Forward Melting Temperature (Tm)'] = f_tm
            row['Reverse Melting Temperature (Tm)'] = r_tm
            row['Tm difference'] = abs(f_tm - r_tm)
            valid_primers.append(row)

        # Create a new DataFrame with valid primers and save to csv
        valid_primers_df = pd.DataFrame(valid_primers)
        valid_primers_df.to_csv(output_file, index=False)

        print(f'Finished processing {primer_type} primers!')
        print(f'Retained {len(valid_primers_df)} out of {len(primers)} primers.')

    print("\nAll primers processed successfully.")


def copy_primer_results(basic_path, variant_list):
    final_result_dir = os.path.join(basic_path, 'Final_result')
    os.makedirs(final_result_dir, exist_ok=True)

    for variant_name in variant_list:
        print(f"Copying results for {variant_name} variant...")

        for primer_type in ['exist', 'new']:
            source_file = os.path.join(basic_path, variant_name, 'reverse', 'amplicon_length',
                                       f'r_{primer_type}_primers_deltaG.csv')
            dest_file = os.path.join(final_result_dir,
                                     f'{primer_type}_{variant_name}_primers_result.csv')

            if os.path.exists(source_file):
                shutil.copyfile(source_file, dest_file)
                print(f"Copied {primer_type} primers result for {variant_name}")
            else:
                print(f"Warning: {primer_type} primers result file for {variant_name} not found")

        print(f"Finished copying results for {variant_name} variant\n")


if __name__ == '__main__':
    basic_path = './'

    variant_list = ['alpha', 'beta', 'gamma', 'delta', 'omicron']  # Can change the order

    for variant_Name in variant_list:
        print(f"Processing {variant_Name} variant...")
        path = os.path.join(basic_path, variant_Name, 'reverse')
        os.makedirs(path, exist_ok=True)

        seq_file = os.path.join(basic_path, 'seq_and_seqName', f'seq_GISAID_{variant_Name}.csv')

        # Load sequence data
        try:
            sequence_full = pd.read_csv(seq_file, header=None).values.ravel()
            print(f"Loaded {len(sequence_full)} sequences for {variant_Name}")
        except FileNotFoundError:
            print(f"Warning: Sequence file for {variant_Name} not found. Skipping this variant.")
            continue


        get_available_primers(path)
        length_of_amplicon(path, sequence_full)
        reshape_file(path)

        reverse(path)
        primer_design_rules(path)

    copy_primer_results(basic_path, variant_list)