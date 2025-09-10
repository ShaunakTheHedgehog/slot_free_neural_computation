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
from utils import *
from mhn_tf import run_case_sequence_model_sweep, OneWinnerMHNLayer, train_mhn_tf_model_batchmode
from baseline_tf import SimplifiedTransformerLayer, train_tf_batchmode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(device)

# generic main function for training and evaluating models
def main():

    ntrials = 10
    model_type = 'mhn_tf_V1'
    num_letters = 4
    full_seq_len = 5
    k_dim = 512
    tf_dim = 10
    debug_mode = True
    criterion = mse_loss
    num_batches = 1_000
    batch_size = 16
    lr = 1e-2
    K_grad_type = 'supervised'
    WV_train_mode = 'via_reinstatement'
    input_dim = 3 * num_letters
    output_dim = 2
    dataset_params = ['case_sequence', num_letters]
    # save_dir = ...


    # code for training and evaluating a single MHN-transformer model (in batch mode)
    model = OneWinnerMHNLayer(batch_size, input_dim, k_dim, output_dim, tf_dim,
                              debug_mode=debug_mode, device=device).to(device)

    Q_losses, K_losses, V_losses, accs, wv, ul_cov, qk_submat = train_mhn_tf_model_batchmode(model, full_seq_len, dataset_params, criterion,
                                 num_batches=num_batches, batch_size=batch_size, lr=lr,
                                 freeze_K=False, freeze_Q=False, freeze_V=False,
                                 manual_grad_calc=True, plot_mode=True, full_key_covar=True,
                                 device=device, K_grad_type=K_grad_type, WV_train_mode=WV_train_mode)
    
    plt.figure()
    plt.plot(accs)
    plt.xlabel('Batch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy across Batches')
    plt.show()

    plt.figure()
    plt.plot(Q_losses, label='Q Loss')
    plt.plot(K_losses, label='K Loss')
    plt.plot(V_losses, label='V Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Q, K, V Losses across Batches')
    plt.legend()
    plt.show()

    # visualize learned Q, K, V weights and covariance matrices
    _, _ = visualize_QKV_matrices(model, 'MHN-tf', label='Final Learned Weights', plot_mode=True, W_V_lims=[-0.2, 1.2, 0.2], QK_lims=[-2, 5, 1])


    # code for training and evaluating a single simplified transformer model (in batch mode)
    # model = SimplifiedTransformerLayer(input_dim, k_dim, 1, output_dim)
    
    # batch_losses, batch_accs, wv, ul_cov, qk_submat = train_tf_batchmode(model, full_seq_len, dataset_params, criterion,
    #                    num_batches=num_batches, batch_size=batch_size, lr=lr,
    #                    freeze_K=True, freeze_Q=True, freeze_V=False, manual_grad_calc=False,
    #                    device=device, plot_mode=False, permutation_reduced=True)
    

    # code for running multiple independent trials of MHN-transformer model training and evaluation (in batch mode)
    # results_dict, all_accs, all_losses, covar_stats_dict, mean_covar_stats_dict = run_case_sequence_model_sweep(ntrials, model_type, num_letters, full_seq_len, k_dim, tf_dim,
    #                                                 debug_mode, criterion, num_batches, batch_size, lr, manual_grad_calc=True,
    #                                                 K_lr=None, K_grad_type=K_grad_type, final_window=500, device=device, save_dir=save_dir, WV_train_mode=WV_train_mode)

    return 

if __name__=="__main__":
    main()