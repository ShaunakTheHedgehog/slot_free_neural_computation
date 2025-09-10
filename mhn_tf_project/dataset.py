import numpy as np
import matplotlib.pyplot as plt
import pdb

import functools
import random
import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset


# case type prediction dataset
# first 'num_letters' tokens are uppercase, second 'num_letters' tokens are lowercase, third 'num_letters' tokens are for querying
def generate_letter_config(num_letters):
    '''
    returns: 
        int_to_letters: shape (num_letters, 2, 3*num_letters), mapping integer letter index to one-hot encoding of upper/lower case letter
        int_to_query_token: shape (num_letters, 3*num_letters), mapping integer letter index to one-hot encoding of query token
    '''
    int_to_letters = torch.zeros(num_letters, 2, 3*num_letters)
    int_to_query_token = torch.zeros(num_letters, 3*num_letters)
    for i in range(num_letters):
        int_to_letters[i, 0] = F.one_hot(torch.tensor([i]), num_classes=3*num_letters)
        int_to_letters[i, 1] = F.one_hot(torch.tensor([num_letters+i]), num_classes=3*num_letters)
        int_to_query_token[i] = F.one_hot(torch.tensor([i+2*num_letters]), num_classes=3*num_letters)

    return int_to_letters, int_to_query_token


# generates case sequence dataset of shape (num_samples, seq_len + 1, 3*num_letters)
def generate_case_sequences(num_samples, seq_len, num_letters):
    '''
    Generates a dataset of sequences of letters with varying cases, along with query tokens.
    Each sequence consists of 'seq_len' letters (each letter can be either uppercase or lowercase),
    followed by a query token that asks for the case of one of the letters in the sequence

    Arguments:
    num_samples : int : number of sequences to generate
    seq_len : int : length of each sequence (number of letters before the query token)
    num_letters : int : number of distinct letters (e.g., if num_letters=3, letters are A, B, C)

    Returns:
    inputs : torch.FloatTensor : shape (num_samples, seq_len + 1, 3*num_letters) : one-hot encoded input sequences with query tokens
    targets : torch.FloatTensor : shape (num_samples, 2) : one-hot encoded target cases (0 for uppercase, 1 for lowercase)
    '''
    assert seq_len <= num_letters

    int_to_letters, int_to_query_token = generate_letter_config(num_letters)

    one_hot_seqs = torch.zeros(num_samples, seq_len, 3*num_letters)
    queries = torch.zeros((num_samples, 3*num_letters))
    targets = torch.zeros(num_samples)

    letter_inds = torch.stack([torch.randperm(num_letters)[:seq_len] for _ in range(num_samples)])
    case_inds = torch.randint(0, 2, size=(num_samples, seq_len))

    for i in range(num_samples):
        for j in range(seq_len):
            one_hot_seqs[i, j] = int_to_letters[letter_inds[i, j], case_inds[i, j]]

    query_positions = torch.randint(0, seq_len, size=(num_samples,))

    targets = torch.gather(case_inds, 1, query_positions.view(-1, 1))
    targets = targets.reshape(-1)
    targets = F.one_hot(targets, num_classes=2)

    query_inds = torch.gather(letter_inds, 1, query_positions.view(-1, 1))

    for k in range(num_samples):
        queries[k] = int_to_query_token[query_inds[k]]
    queries = queries.unsqueeze(1)

    inputs = torch.cat((one_hot_seqs, queries), 1)
    targets = targets.type(torch.FloatTensor)

    return inputs, targets

# generates a small toy case sequence dataset (with 2 letters, A and B) for debugging purposes
# 'reduced' flag generates a smaller dataset with fewer sequences, precisely avoiding permutation-based duplicates (e.g. A, B and B, A)
def generate_toy_case_sequence_dataset(reduced=False):
    int_to_letters, int_to_query_token = generate_letter_config(2)
    inputs, outputs = None, None

    if not reduced:
        inputs = torch.zeros(8, 3, 6)
        inputs[0, 0] = int_to_letters[0, 0]
        inputs[1, 0] = int_to_letters[0, 0]
        inputs[4, 0] = int_to_letters[0, 0]
        inputs[5, 0] = int_to_letters[0, 0]

        inputs[0, 1] = int_to_letters[1, 0]
        inputs[1, 1] = int_to_letters[1, 0]
        inputs[2, 1] = int_to_letters[1, 0]
        inputs[3, 1] = int_to_letters[1, 0]

        inputs[2, 0] = int_to_letters[0, 1]
        inputs[3, 0] = int_to_letters[0, 1]
        inputs[6, 0] = int_to_letters[0, 1]
        inputs[7, 0] = int_to_letters[0, 1]

        inputs[4, 1] = int_to_letters[1, 1]
        inputs[5, 1] = int_to_letters[1, 1]
        inputs[6, 1] = int_to_letters[1, 1]
        inputs[7, 1] = int_to_letters[1, 1]

        for i in range(4):
            inputs[2*i, 2] = int_to_query_token[0]
            inputs[2*i+1, 2] = int_to_query_token[1]

        outputs = torch.FloatTensor([[1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])

    else:
        inputs = torch.zeros(4, 3, 6)

        inputs[0, 0] = int_to_letters[0, 0]
        inputs[1, 0] = int_to_letters[0, 0]

        inputs[2, 1] = int_to_letters[1, 0]
        inputs[3, 1] = int_to_letters[1, 0]

        inputs[2, 0] = int_to_letters[0, 1]
        inputs[3, 0] = int_to_letters[0, 1]

        inputs[0, 1] = int_to_letters[1, 1]
        inputs[1, 1] = int_to_letters[1, 1]

        for i in range(2):
            inputs[2*i, 2] = int_to_query_token[0]
            inputs[2*i+1, 2] = int_to_query_token[1]

        outputs = torch.FloatTensor([[1, 0], [0, 1], [0, 1], [1, 0]])

    return inputs, outputs


# helper function to check if a vector is present in a tensor along the dimension dim = 1
def is_present(vector, tensor):
    return torch.any(torch.all(tensor == vector, dim=1))


# generates a reduced dataset for the case sequence task, where the sequences are not permuted
# i.e. A, B and B, A are considered the same sequence
def generate_permutation_reduced_dataset(num_letters):
    ''' 
    Returns a dataset (inputs, targets), with 'inputs' having size [total # of permutation-reduced sequences] x seq_len x input_dim
    and 'targets' having size [total # of permutation-reduced sequences] x 2.
    Here, seq_len = num_letters + 1 and input_dim = 3 * num_letters.
    '''
    input_dim = 3 * num_letters
    sets = []

    for i in range(num_letters):
        lower_upper_letters = F.one_hot(torch.tensor([i, num_letters + i]), num_classes=input_dim)
        sets.append(lower_upper_letters)

    queries = F.one_hot(torch.arange(2*num_letters, 3*num_letters), num_classes=input_dim)
    sets.append(queries)

    combinations = itertools.product(*sets)

    # Convert combinations to a list of tensors
    combinations_list = [torch.stack(combination) for combination in combinations]

    # Stack into a single tensor
    inputs = torch.stack(combinations_list)
    inputs = inputs.float()

    targets = []
    for j in range(inputs.shape[0]):
        curr_seq = inputs[j]
        context = curr_seq[:-1, :]
        query = curr_seq[-1]

        one_idx = torch.argmax(query)
        lower_idx = one_idx - 2 * num_letters
        upper_idx = one_idx - num_letters

        lower = torch.zeros(input_dim)
        lower[lower_idx] = 1

        upper = torch.zeros(input_dim)
        upper[upper_idx] = 1

        if is_present(lower, context):
            targets.append(torch.tensor([1, 0]))
        else:
            assert is_present(upper, context)
            targets.append(torch.tensor([0, 1]))

    targets = torch.stack(targets)
    targets = targets.float()

    return inputs, targets