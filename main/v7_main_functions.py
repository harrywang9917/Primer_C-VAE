# Description: This file contains the main functions used in the pipeline
# Update: 2024/04/09


from concurrent.futures import ProcessPoolExecutor
import os



def sys_path(base_path):
    variant_list = ['alpha', 'beta', 'gamma', 'delta', 'omicron']
    general_folders = ['Seq_and_SeqName', 'index', 'model', 'Final_result']
    forward_reverse_folders = ['filter', 'posPool', 'maxPool', 'filter_seq', 'featsVector', 'dataDNAFeatures', 'feature',
                               'Seq_appearance', 'result']
    reverse_additional_folders = ['model', 'amplicon_length', 'CG_Check', 'exist_primer_check', 'forward_primer',
                                  'second_data']
    cg_check_sub_folders = ['new_primers', 'exist_primers']

    # Create general folders not specific to any variant
    for folder in general_folders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

    for variant in variant_list:
        GISAID_path = os.path.join(base_path, 'Variant_virus', 'GISAID_' + variant)
        variant_path = os.path.join(base_path, variant)
        os.makedirs(GISAID_path, exist_ok=True)  # Create variant specific folder

        for direction in ['forward', 'reverse']:
            for folder in forward_reverse_folders:
                os.makedirs(os.path.join(variant_path, direction, folder), exist_ok=True)

            if direction == 'reverse':
                for folder in reverse_additional_folders:
                    specific_folder_path = os.path.join(variant_path, direction, folder)
                    os.makedirs(specific_folder_path, exist_ok=True)

                    if folder == 'CG_Check':
                        for sub_folder in cg_check_sub_folders:
                            os.makedirs(os.path.join(specific_folder_path, sub_folder), exist_ok=True)


def readFASTA(fa):
    '''
    Read a FASTA file and return a dictionary with sequence names as keys and sequences as values.

    :param fa: str, the path of the FASTA file
    :return: dict, a dictionary with key = seqName and value = sequence
    '''
    seqDict = {}
    with open(fa, 'r') as FA:
        seqName = ''  # Initialize seqName to ensure it's defined before use
        for line in FA:
            line = line.strip()  # Strip whitespace once per line
            if line.startswith('>'):
                seqName = line[1:].split()[0]
                seqDict[seqName] = ''
            else:
                seqDict[seqName] += line
    return seqDict


def readFASTA_iter(fa):
    '''
    Read a FASTA file and yield each sequence name and sequence.

    :param fa: str, the path of the FASTA file
    :return: generator, yields a tuple (sequence name, sequence) for each sequence in the FASTA file
    '''
    seqName, seq = None, []
    with open(fa, 'r') as FA:
        for line in FA:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if seqName is not None:  # Yield previous sequence before starting a new one
                    yield (seqName, ''.join(seq))
                seqName = line[1:].split()[0]  # Start new sequence name
                seq = []  # Reset sequence list
            else:
                seq.append(line)  # Build sequence list
        if seqName is not None:  # Yield the last sequence in the file
            yield (seqName, ''.join(seq))


def getSeq(fa, querySeqName, start=1, end=None):
    '''
    Get a particular sequence from a FASTA file by sequence name with optional slicing.

    :param fa: str, the path of the FASTA file
    :param querySeqName: str, the name of the sequence to fetch
    :param start: int, the starting position for slicing the sequence (1-indexed, default is 1)
    :param end: int, the ending position for slicing the sequence (None for full length, inclusive)
    :return: str, the sliced sequence if found, otherwise an empty string
    '''
    if start < 1:
        start = 1  # Ensure start is at least 1, adhering to 1-based indexing outside Python

    for seqName, seq in readFASTA_iter(fa):
        if querySeqName == seqName:
            # If end is not specified (None), slice to the end of the sequence
            # Adjust for Python's 0-based indexing by subtracting 1 from start
            return seq[start - 1:end] if end is not None else seq[start - 1:]

    return ""  # Return an empty string if the sequence name is not found


def getReverseComplement(sequence):
    '''
    Get the reverse complementary DNA (cDNA) of an RNA sequence.

    :param sequence: str, an RNA sequence of the virus
    :return: str, the reverse cDNA sequence of the given RNA
    '''
    sequence = sequence.upper()
    complement = {'A': 't', 'T': 'q', 'G': 'c', 'C': 'g'}
    # Replace each nucleotide with its complement and reverse the sequence
    reverse_complement = ''.join(complement[nuc] for nuc in sequence[::-1])
    return reverse_complement


def getGC(sequence):
    '''
    Get the GC content of a sequence.

    :param sequence: str, a sequence of RNA
    :return: float, the GC content of the sequence as a fraction of the total sequence length
    '''
    sequence = sequence.upper()
    total_length = len(sequence)
    if total_length == 0:
        return 0.0  # Optionally, could raise ValueError("Sequence is empty.") for stricter handling
    gc_content = (sequence.count("G") + sequence.count("C")) / total_length
    return gc_content


def calculateGCContents(sequences):
    '''
    Calculate the GC content of multiple sequences in parallel using a process pool.

    :param sequences:
    :return:
    '''
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(getGC, sequences))
    return results


def readSeqByWindow(sequence, winSize, stepSize):
    '''
    Use a sliding window to read subsequences of a given RNA sequence.

    :param sequence: str, the RNA sequence to be processed
    :param winSize: int, the window size
    :param stepSize: int, the step size for the sliding window
    :return: generator, yields each subsequence captured by the window
    '''
    if stepSize <= 0:
        raise ValueError("Step size must be positive.")  # Consider raising an exception for robust error handling

    now = 0
    seqLen = len(sequence)
    while now + winSize <= seqLen:  # Ensure all subsequences, including the last possible window, are considered
        yield sequence[now:now + winSize]
        now += stepSize

