# This file contains the functions that are used in the main_model.py file
# Update: 2024/04/09

import numpy as np
import pandas as pd
import random
import torch

# Check if CUDA is available and set the default device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data):
    """
    Moves tensor to the defined device.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(data, (list, tuple)):
        return [to_device(x) for x in data]
    return data.to(device)


def oneHot_pytorch(array, size):
    return torch.nn.functional.one_hot(torch.tensor(array).to(torch.int64), num_classes=size).to(device)


def weight_variable_pytorch(shape):
    return torch.nn.Parameter(torch.randn(shape, device=device) * 0.1)


def bias_variable_pytorch(shape):
    return torch.nn.Parameter(torch.ones(shape, device=device) * 0.1)


def getBatch_run_pytorch(data, labels, size, run, vector, sampleSize):
    infLimit = run * size
    supLimit = infLimit + size
    if supLimit > len(data):
        supLimit = len(data)
    batch = vector[infLimit:supLimit]
    outData = torch.zeros(len(batch), sampleSize, device=device)
    outLabels = torch.zeros(len(batch), dtype=torch.long, device=device)
    for i, idx in enumerate(batch):
        for j, nucleotide in enumerate(data[idx]):
            outData[i, j] = {'C': 0.25, 'T': 0.5, 'G': 0.75, 'A': 1.0}.get(nucleotide, 0.0)
        outLabels[i] = labels[idx]
    return outData, outLabels


def getBatch_pytorch(data, labels, size, sampleSize):
    batch = random.sample(range(len(data)), size)
    outData = torch.zeros(size, sampleSize, device=device)
    outLabels = torch.zeros(size, dtype=torch.long, device=device)
    for i, idx in enumerate(batch):
        for j, nucleotide in enumerate(data[idx]):
            outData[i, j] = {'C': 0.25, 'T': 0.5, 'G': 0.75, 'A': 1.0}.get(nucleotide, 0.0)
        outLabels[i] = labels[idx]
    return outData, outLabels


def getBatch_pytorch(data, size, sampleSize):
    batch = list(range(len(data)))
    outData = torch.zeros(size, sampleSize, device=device).cpu().numpy()
    for i, idx in enumerate(batch):
        for j, nucleotide in enumerate(data[idx]):
            outData[i, j] = {'C': 0.25, 'T': 0.5, 'G': 0.75, 'A': 1.0}.get(nucleotide, 0.0)
    return torch.tensor(outData, device=device)
