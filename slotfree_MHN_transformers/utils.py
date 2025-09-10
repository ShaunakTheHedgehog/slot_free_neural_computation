import numpy as np
import matplotlib.pyplot as plt
import pdb

import functools
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from einops import rearrange, repeat

from tqdm.auto import trange
from copy import deepcopy
import warnings
import torch.optim as optim
from torch.optim import lr_scheduler

# General helper functions
#---------------------------------------------------------------------

# k-winner-take-all function that sets the largest 'k' entries 
# of the vector 'x' to 1 and the rest to 0
def kWTA(x, k):
  sparse = torch.zeros_like(x)
  max_inds = torch.argsort(x.reshape(-1))
  max_inds = max_inds[-k:]
  sparse[max_inds] = 1
  return sparse


# softmax function with temperature scaling
def softmax(input, temperature=1.):
    sigma = nn.Softmax(dim=-1)
    return sigma(input/temperature)


# apply a k-winner-take-all function to a tensor 'x' along dimension 'dim'
def sparsify_tensor(x, k, dim=-1):
    # Find the indices of the largest 3 entries along the last dimension
    _, indices = torch.topk(x, k=k, dim=dim)

    sparse_x = torch.zeros_like(x)

    # Set the indices of the largest k entries to 1
    sparse_x.scatter_(dim, indices, 1)

    return sparse_x

# plot training losses and accuracies
def plot_loss_acc(losses, accs, label='', title_label='', download=False):
    plt.figure()
    plt.ylabel('Loss', fontsize=17)
    plt.xlabel('Iteration', fontsize=17)
    plt.ylim(bottom=0)
    plt.plot(np.asarray(losses))
    plt.tick_params(axis='both', which='major', labelsize=17)  # Change the size of major ticks
    plt.title('Batch MSE Loss across Gradient Steps', fontsize=18.5)
    if download:
        plt.savefig(label + '_losses.pdf', bbox_inches='tight')


    plt.figure()
    plt.xlabel('Iteration', fontsize=17)
    plt.ylabel('Accuracy', fontsize=17)
    plt.title(title_label + 'Batch Accuracy across Gradient Steps', fontsize=18.5)
    plt.ylim(0, 1.05)
    plt.tick_params(axis='both', which='major', labelsize=17)  # Change the size of major ticks
    plt.plot(np.asarray(accs))
    if download:
        plt.savefig(label + '_accs.pdf', bbox_inches='tight')


# cross-entropy loss function
def ce_loss(model_outputs, target_labels):
    # both 'model_outputs' and 'target_labels' should be batch_size x 2
    if model_outputs.dim() == 2:
        model_outputs = model_outputs.unsqueeze(-2)
    query_outputs = model_outputs[:, -1, :]
    return F.cross_entropy(query_outputs, target_labels)


# mean-squared error loss function
def mse_loss(model_outputs, target_labels):
    if model_outputs.dim() == 2:
        model_outputs = model_outputs.unsqueeze(-2)
    model_outputs = model_outputs[:, -1, :]
    batch_size = len(target_labels)
    return torch.sum((model_outputs - target_labels)**2) # * (1. / batch_size)


# accuracy calculation function based on model outputs and target labels
def get_acc(val_outputs, val_labels):
    # both 'val_outputs' and 'val_labels' should be batch_size x v_dim
    if val_outputs.dim() == 2:
        val_outputs = val_outputs.unsqueeze(-2)
    logits = val_outputs[:, -1, :]
    acc = 1.*torch.sum((torch.argmax(logits, axis=-1) == torch.argmax(val_labels, axis=-1))) / len(logits)
    return acc


lowercase_alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

uppercase_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

abstract_alphabet = ['a/A', 'b/B', 'c/C', 'd/D', 'e/E', 'f/F', 'g/G', 'h/H', 'i/I', 'j/J', 'k/K', 'l/L', 'm/M',
                     'n/N', 'o/O', 'p/P', 'q/Q', 'r/R', 's/S', 't/T', 'u/U', 'v/V', 'w/W', 'x/X', 'y/Y', 'z/Z']

cases = ['lowercase', 'uppercase']

# get x tick labels for plotting similarity matrices for different letter types
def get_x_labels(input_size, version='all'):
    num_letters = int(input_size / 3)
    x_ticks = np.arange(num_letters)
    x_labels_l = [lowercase_alphabet[i] for i in x_ticks]  # Custom x tick labels
    if version=='lowercase':
        return x_labels_l
    x_labels_u = [uppercase_alphabet[i] for i in x_ticks]
    if version=='uppercase':
        return x_labels_u
    x_labels_abs = [abstract_alphabet[i] for i in x_ticks]
    if version=='abstract':
        return x_labels_abs
    if version=='both_cases':
        return x_labels_l + x_labels_u
    x_labels = x_labels_l + x_labels_u + x_labels_abs
    return x_labels


# visualize Q, K, V matrices from a given attention-based model and their products
def visualize_QKV_matrices(model, model_type, label='', download=False, plot_mode=True, W_V_lims=None, QK_lims=None):
    assert model_type == 'tf' or model_type == 'mhn_tf'

    W_Q, W_K, W_V = None, None, None

    if model_type == 'tf':
        W_Q, W_K, W_V = model.self_attn.q_project.weight.data, model.self_attn.k_project.weight.data, model.self_attn.v_project.weight.data
    else:
        W_Q, W_K, W_V = model.W_Q.weight.data, model.W_K.weight.data, model.W_V.weight.data

    W_Q = W_Q.detach().cpu().numpy()
    W_K = W_K.detach().cpu().numpy()
    W_V = W_V.detach().cpu().numpy()

    if plot_mode:
        plt.figure()
        plt.title('$W_Q$ ' + label)
        plt.imshow(W_Q)
        plt.colorbar()

        plt.figure()
        plt.title('$W_K$ ' + label)
        plt.imshow(W_K)
        plt.colorbar()

        plt.figure(figsize=(8, 6))
        plt.title('Entries of $W_V$', fontsize=20)
        vmin = W_V.min()
        vmax = W_V.max()
        if W_V_lims is not None:
            vmin = W_V_lims[0]
            vmax = W_V_lims[1]
        plt.imshow(W_V, cmap='viridis', vmin=vmin, vmax=vmax)
        # Set custom x and y ticks
        x_labels = get_x_labels(W_V.shape[1])
        y_labels = cases  # Custom y tick labels

        plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, fontsize=17)  # Custom x ticks and font size
        plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels, fontsize=17)  # Custom y ticks and font size

        # Add colorbar and adjust its font size
        cbar = plt.colorbar(orientation='horizontal')

        # Set specific ticks on the colorbar
        if W_V_lims is not None:
            cbar.set_ticks(np.arange(W_V_lims[0], W_V_lims[1]+0.001, W_V_lims[2]))
        cbar.ax.tick_params(labelsize=17)

        if download:
            plt.savefig(label + '_WV.pdf', bbox_inches='tight')



    qk_prod = W_Q.T @ W_K

    num_letters = int(W_K.shape[1]/3.)
    qk_submatrix = qk_prod[2*num_letters:, :2*num_letters]

    if plot_mode:
        plt.figure(figsize=(10, 5))
        plt.title('Entries of $W_Q^T W_K$', fontsize=20)
        vmin = qk_submatrix.min()
        vmax = qk_submatrix.max()
        if QK_lims is not None:
            vmin = QK_lims[0]
            vmax = QK_lims[1]
        plt.imshow(qk_submatrix, cmap='viridis', vmin=vmin, vmax=vmax)
        # Set custom x and y ticks
        x_labels = get_x_labels(W_V.shape[1], version='both_cases')
        y_labels = get_x_labels(W_V.shape[1], version='abstract')

        plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, fontsize=20)  # Custom x ticks and font size
        plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels, fontsize=20)  # Custom y ticks and font size

        # Add colorbar and adjust its font size
        cbar = plt.colorbar(orientation='horizontal')
        if QK_lims is not None:
            cbar.set_ticks(np.arange(QK_lims[0], QK_lims[1]+0.001, QK_lims[2]))
        cbar.ax.tick_params(labelsize=20)

        if download:
            plt.savefig(label + '_QK_cov.pdf', bbox_inches='tight')

    return W_V, qk_submatrix


# visualize covariance between the keys for lowercase and uppercase letters
def visualize_uppercase_lowercase_covariance(num_letters, W_K, label='', download=False, plot_mode=True, KK_lims=None, full=False):
    W_K = W_K.detach().cpu().numpy()

    letters = np.zeros((2*num_letters, 3*num_letters))
    letters[:, :2*num_letters] = torch.eye(2*num_letters)
    lowers = letters[:num_letters]                              # num_letters x input_dim
    uppers = letters[num_letters:]                              # num_letters x input_dim

    lower_ks = lowers @ W_K.T                                   # num_letters x k_dim
    upper_ks = uppers @ W_K.T                                   # num_letters x k_dim

    covar = lower_ks @ upper_ks.T                               # num_letters (lowercase) x num_letters (uppercase)

    if full:
        ks = letters @ W_K.T                                    # (2*num_letters) x k_dim
        covar = ks @ ks.T
    # print(covar)

    if plot_mode and not full:
        plt.figure(figsize=(7, 7))
        plt.title('Uppercase-Lowercase \nKey Covariance', fontsize=19)
        vmin = covar.min()
        vmax = covar.max()
        if KK_lims is not None:
            vmin = KK_lims[0]
            vmax = KK_lims[1]
        plt.imshow(covar, cmap='viridis', vmin=vmin, vmax=vmax)
        x_labels = get_x_labels(W_K.shape[1], version='uppercase')
        y_labels = get_x_labels(W_K.shape[1], version='lowercase')  # Custom y tick labels

        plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, fontsize=20)  # Custom x ticks and font size
        plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels, fontsize=20)  # Custom y ticks and font size

        # Add colorbar and adjust its font size
        cbar = plt.colorbar(orientation='horizontal')
        if KK_lims is not None:
            cbar.set_ticks(np.arange(KK_lims[0], KK_lims[1]+0.001, KK_lims[2]))
        cbar.ax.tick_params(labelsize=18)

        if download:
            plt.savefig(label + '_upperlower_covariance.pdf', bbox_inches='tight')

    if plot_mode and full:
        plt.figure(figsize=(10, 10))
        plt.title('Key Covariance', fontsize=20)
        plt.imshow(covar, cmap='viridis', vmin=KK_lims[0], vmax=KK_lims[1])
        x_labels = get_x_labels(W_K.shape[1], version='both_cases')
        y_labels = get_x_labels(W_K.shape[1], version='both_cases')

        plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, fontsize=20)  # Custom x ticks and font size
        plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels, fontsize=20)  # Custom y ticks and font size

        # Add colorbar and adjust its font size
        cbar = plt.colorbar(orientation='horizontal')
        if KK_lims is not None:
            cbar.set_ticks(np.arange(KK_lims[0], KK_lims[1]+0.001, KK_lims[2]))
        cbar.ax.tick_params(labelsize=20)

        if download:
            plt.savefig(label + '_upperlower_covariance.pdf', bbox_inches='tight')

    return covar


# get averaged difference between entries of W_V corresponding to 1s and 0s in the output
def get_W_V_stats(num_letters, W_V):
    ones = np.sum(W_V[0, :num_letters] + W_V[1, num_letters:2*num_letters]) / (2 * num_letters)
    zeros = np.sum(W_V[0, num_letters:2*num_letters] + W_V[1, :num_letters]) / (2 * num_letters)
    return ones - zeros


# get diagonal and off-diagonal stats of the query-key covariance matrix (W_Q^T W_K)
def get_QK_covar_stats(num_letters, QK_covar):
    mask = 1. - np.eye(num_letters)
    cov_l = QK_covar[:, :num_letters]
    cov_u = QK_covar[:, num_letters:2*num_letters]

    l_diags = np.diag(cov_l)
    l_diags_range = l_diags.max() - l_diags.min()
    l_diags_mean = l_diags.mean()

    l_offdiags = cov_l[mask == 1]
    l_offdiags_range = l_offdiags.max() - l_offdiags.min()
    l_offdiags_mean = l_offdiags.mean()

    u_diags = np.diag(cov_u)
    u_diags_range = u_diags.max() - u_diags.min()
    u_diags_mean = u_diags.mean()

    u_offdiags = cov_u[mask == 1]
    u_offdiags_range = u_offdiags.max() - u_offdiags.min()
    u_offdiags_mean = u_offdiags.mean()

    return l_diags_mean, l_diags_range, l_offdiags_mean, l_offdiags_range, u_diags_mean, u_diags_range, u_offdiags_mean, u_offdiags_range


# get diagonal and off-diagonal stats of the uppercase-lowercase key covariance matrix (W_K^T W_K)
def get_upper_lower_covar_stats(num_letters, ul_covar):
    mask = 1. - np.eye(num_letters)
    diags = np.diag(ul_covar)

    diags_range = diags.max() - diags.min()
    diags_mean = diags.mean()

    offdiags = ul_covar[mask == 1]
    offdiags_range = offdiags.max() - offdiags.min()
    offdiags_mean = offdiags.mean()

    return diags_mean, diags_range, offdiags_mean, offdiags_range



# Manual gradient calculation routines for Transformer
#---------------------------------------------------------------------

# calculate gradients for W_Q matrix in a minimal Transformer model
def calculate_Q_grad(batch_size, y_hat, y, W_Q, W_K, W_V, X):
    '''
    Computes gradient for W_Q for a batch of data in minimal Transformer model

    batch_size   : size of a gradient batch
    y_hat        : output of Transformer model (batch_size x v_dim)
    y            : ground truth output (batch_size x v_dim)
    W_Q          : query matrix (k_dim x input_dim)
    W_K          : key matrix (k_dim x input_dim)
    W_V          : value matrix (v_dim x input_dim)
    X            : inputs (batch_size x seq_len x input_dim)
    '''

    context_len = X.shape[1] - 1
    Q_grad = torch.zeros(W_Q.shape[0], X.shape[-1])

    # add up gradients from each context sequence in the batch
    for i in range(batch_size):
        X_c = X[i, :-1, :]
        x_q = X[i, -1, :]

        e = y_hat[i] - y[i]
        V = W_V @ X_c.T         # v_dim x (seq_len - 1)
        K = W_K @ X_c.T         # k_dim x (seq_len - 1)
        q = W_Q @ x_q

        w = K.T @ q
        sm = F.softmax(w)
        J_sm = torch.diag(sm) - torch.outer(sm, sm)

        first_term = e.reshape(1,-1) @ V @ J_sm
        first_term = first_term.reshape(-1)             # context_len

        K_tensor = K.T.unsqueeze(-1)                                            # context_len x k_dim x 1
        x_q_tensor = x_q.unsqueeze(0).repeat(context_len, 1).unsqeeze(1)        # context_len x 1 x input_dim
        big_tensor = torch.bmm(K_tensor, x_q_tensor)                            # context_len x k_dim x input_dim

        grad = torch.einsum('i,ijk->jk', first_term, big_tensor)
        Q_grad += grad

    Q_grad *= (2./ batch_size)
    return Q_grad


# calculate gradients for W_V matrix in a minimal Transformer model
def calculate_V_grad(batch_size, y_hat, y, W_Q, W_K, W_V, X, x_reinst=None):
    '''
    Computes gradient for W_V for a batch of data in minimal Transformer model

    batch_size   :     size of a gradient batch
    y_hat        :     output of Transformer model (batch_size x v_dim)
    y            :     ground truth output (batch_size x v_dim)
    W_Q          :     query matrix (k_dim x input_dim)
    W_K          :     key matrix (k_dim x input_dim)
    W_V          :     value matrix (v_dim x input_dim)
    X            :     inputs (batch_size x seq_len x input_dim)
    x_reinst     :     reinstated context vectors (batch_size x input_dim)
    '''
    error = y_hat - y          # output error; batch_size x v_dim

    # if the softmax-weigted sum of the x's is provided, use it to compute the gradient more cleanly
    if x_reinst is not None:
        return (2./batch_size) * error.T @ x_reinst

    V_grad = torch.zeros(y.shape[0], X.shape[-1])

    # add up gradients from each context sequence in the batch
    for i in range(batch_size):
        X_c = X[i, :-1, :]      # context_len x input_dim
        x_q = X[i, -1, :]

        V = W_V @ X_c.T         # v_dim x context_len
        K = W_K @ X_c.T         # k_dim x context_len
        q = W_Q @ x_q           # k_dim

        w = K.T @ q             # context_len
        sm = F.softmax(w)

        x_tilde = X_c.T @ sm

        V_grad += torch.outer(error[i], x_tilde)

    V_grad *= (2. / batch_size)
    return V_grad


# calculate gradients for W_K matrix in a minimal Transformer model
def calculate_K_grad(batch_size, y_hat, y, W_Q, W_K, W_V, X):
    '''
    Computes gradient for W_K for a batch of data in minimal Transformer model

    batch_size   : size of a gradient batch
    y_hat        : output of Transformer model (batch_size x v_dim)
    y            : ground truth output (batch_size x v_dim)
    W_Q          : query matrix (k_dim x input_dim)
    W_K          : key matrix (k_dim x input_dim)
    W_V          : value matrix (v_dim x input_dim)
    X            : inputs (batch_size x seq_len x input_dim)
    x_reinst     : reinstated context vectors (batch_size x input_dim)
    '''

    context_len = X.shape[1] - 1
    K_grad = torch.zeros(W_Q.shape[0], X.shape[-1])

    # add up gradients from each context sequence in the batch
    for i in range(batch_size):
        X_c = X[i, :-1, :]      # context_len x input_dim
        x_q = X[i, -1, :]

        e = y_hat[i] - y[i]
        V = W_V @ X_c.T         # v_dim x context_len
        K = W_K @ X_c.T         # k_dim x context_len
        q = W_Q @ x_q           # k_dim

        w = K.T @ q
        sm = F.softmax(w)
        # compute softmax Jacobian
        J_sm = torch.diag(sm) - torch.outer(sm, sm)

        first_term = e.reshape(1,-1) @ V @ J_sm
        first_term = first_term.reshape(-1)             # context_len

        q_tensor = q.unsqueeze(0).repeat(context_len, 1).unsqueeze(-1)       # context_len x k_dim x 1
        big_tensor = torch.bmm(q_tensor, X_c.unsqueeze(1))                   # context_len x k_dim x input_dim

        grad = torch.einsum('i,ijk->jk', first_term, big_tensor)
        K_grad += grad

    K_grad *= (2./ batch_size)


# compute gradients for W_Q, W_K, W_V matrices in a minimal Transformer model
def calculate_QKV_grads(batch_size, y_hat, y, W_Q, W_K, W_V, X, 
                        normalization=1.0, device=torch.device('cpu')):
    '''
    Computes gradients for W_Q, W_K, W_V for a batch of data in minimal Transformer model

    batch_size    : size of a gradient batch
    y_hat         : output of Transformer model (batch_size x v_dim)
    y             : ground truth output (batch_size x v_dim)
    W_Q           : query matrix (k_dim x input_dim)
    W_K           : key matrix (k_dim x input_dim)
    W_V           : value matrix (v_dim x input_dim)
    X             : inputs (batch_size x seq_len x input_dim)
    normalization : normalization factor for the gradients, if necessary
    device        : device to perform the computations on'''

    context_len = X.shape[1] - 1
    K_grad = torch.zeros(W_Q.shape[0], X.shape[-1]).to(device)
    Q_grad = torch.zeros_like(K_grad).to(device)
    V_grad = torch.zeros(y.shape[-1], X.shape[-1]).to(device)


    error = y_hat - y           # batch_size x v_dim

    # add up gradients from each context sequence in the batch
    for i in range(batch_size):
        X_c = X[i, :-1, :]      # context_len x input_dim
        x_q = X[i, -1, :]

        e = y_hat[i] - y[i]
        V = W_V @ X_c.T         # v_dim x context_len
        K = W_K @ X_c.T         # k_dim x context_len
        q = W_Q @ x_q           # k_dim

        w = K.T @ q
        sm = F.softmax(w, dim=0)
        # compute softmax Jacobian
        J_sm = torch.diag(sm) - torch.outer(sm, sm)

        first_term = e.reshape(1,-1) @ V @ J_sm
        first_term = first_term.reshape(-1)             # context_len

        # calculate W_K gradient
        q_tensor = q.unsqueeze(0).repeat(context_len, 1).unsqueeze(-1)        # context_len x k_dim x 1
        big_tensor_for_K = torch.bmm(q_tensor, X_c.unsqueeze(1))             # context_len x k_dim x input_dim

        little_K_grad = torch.einsum('i,ijk->jk', first_term, big_tensor_for_K)
        K_grad += little_K_grad

        # calculate W_Q gradient
        K_tensor = K.T.unsqueeze(-1)                                            # context_len x k_dim x 1
        x_q_tensor = x_q.unsqueeze(0).repeat(context_len, 1).unsqueeze(1)        # context_len x 1 x input_dim
        big_tensor_for_Q = torch.bmm(K_tensor, x_q_tensor)                      # context_len x k_dim x input_dim

        little_Q_grad = torch.einsum('i,ijk->jk', first_term, big_tensor_for_Q)
        Q_grad += little_Q_grad

        # calculate W_V gradient
        x_tilde = X_c.T @ sm

        V_grad += torch.outer(error[i], x_tilde)

    K_grad *= (2. / normalization)
    Q_grad *= (2. / normalization)
    V_grad *= (2. / normalization)

    return Q_grad, K_grad, V_grad




# Manual gradient calculation routines for MHN-Transformer models
#---------------------------------------------------------------------

# calculate gradients for W_Q and W_V matrices in a MHN-Transformer model
def calculate_mhn_tf_QV_grads(batch_size, y_hat, y, W_Q, W_MHN_tensor, W_reinst_tensor, W_V, X, x_reinst,
                              device=torch.device('cpu'), WV_train_mode='via_reinstatement', MHN_out=None):
    '''
    Computes gradients for W_Q, W_V for a batch of data in the MHN-Transformer model

    batch_size      :   size of a gradient batch
    y_hat           :   output of Transformer model (batch_size x v_dim)
    y               :   ground truth output (batch_size x v_dim)
    W_Q             :   query matrix (k_dim x input_dim)
    W_MHN_tensor    :   MHN weight matrices (batch_size x tf_dim x k_dim)
    W_reinst_tensor :   reinstatement weight matrices (batch_size x tf_dim x input_dim)     
    W_V             :   value matrix (v_dim x input_dim)
    X               :   inputs (batch_size x seq_len x input_dim)
    x_reinst        :   reinstated context vectors (batch_size x input_dim)
    device          :   device to perform the computations on
    WV_train_mode   :   'via_reinstatement' or 'via_MHN_output'
    MHN_out         :   output of the MHN (batch_size x v_dim); required if WV_train_mode=='via_MHN_output'
    '''

    assert WV_train_mode in ['via_reinstatement', 'via_MHN_output']

    # context_len = X.shape[1] - 1
    Q_grad = torch.zeros(W_Q.shape[0], X.shape[-1]).to(device)
    V_grad = torch.zeros(y.shape[-1], X.shape[-1]).to(device)

    error = y_hat - y   
    if WV_train_mode == 'via_reinstatement':                            # batch_size x v_dim
        V_grad = 2. * error.T @ x_reinst
    else:
        assert MHN_out is not None
        V_grad = 2. * (MHN_out - y).T @ x_reinst

    # add up gradients from each context sequence in the batch
    for i in range(batch_size):
        W_MHN = W_MHN_tensor[i]         # tf_dim x k_dim
        W_reinst = W_reinst_tensor[i]   # tf_dim x input_dim
        tf_dim = W_MHN.shape[0]
        X_c = X[i, :-1, :]              # context_len x input_dim
        x_q = X[i, -1, :]               # input_dim

        e = y_hat[i] - y[i]             # v_dim
        q = W_Q @ x_q                   # k_dim

        w = W_MHN @ q
        sm = F.softmax(w, dim=0)
        # compute softmax Jacobian
        J_sm = torch.diag(sm) - torch.outer(sm, sm)

        first_term_Q = e.reshape(1,-1) @ W_V @ W_reinst.T @ J_sm
        first_term_Q = first_term_Q.reshape(-1)         # tf_dim

        # calculate W_Q gradient
        K_tensor = W_MHN.unsqueeze(-1)                                      # tf_dim x k_dim x 1
        x_q_tensor = x_q.unsqueeze(0).repeat(tf_dim, 1).unsqueeze(1)        # tf_dim x 1 x input_dim
        big_tensor_for_Q = torch.bmm(K_tensor, x_q_tensor)                  # tf_dim x k_dim x input_dim

        little_Q_grad = torch.einsum('i,ijk->jk', first_term_Q, big_tensor_for_Q)
        Q_grad += (2 * little_Q_grad)

    return Q_grad, V_grad


# simple supervised or Hebbian update for W_K matrix in a MHN-Transformer model
def simple_K_update(W_K, x_reinst, W_Q, x_q, 
                    update_type='supervised', device=torch.device('cpu')):
    '''
    Computes simple supervised or Hebbian gradient for W_K in the MHN-Transformer model

    W_K          :   key matrix (k_dim x input_dim)
    x_reinst     :   reinstated context vectors (batch_size x input_dim)
    W_Q          :   query matrix (k_dim x input_dim)
    x_q          :   query vector (batch_size x input_dim)
    update_type  :   'supervised' (i.e. supervised query-key alignment) or 'Hebbian'
    device       :   device to perform the computations on
    '''

    assert update_type=='supervised' or update_type=='Hebbian'
    K_grad = torch.zeros_like(W_K).to(device)

    q = x_q @ W_Q.T                               # batch_size x k_dim
    k_tilde = x_reinst @ W_K.T                    # batch_size x k_dim

    if update_type=='supervised':
        K_grad = 2 * (k_tilde - q).T @ x_reinst
    else:
        K_grad = -q.T @ x_reinst

    return K_grad


# calculate backwards gradients for the W_K matrix in a MHN-Transformer model through the MHN module
def calculate_mhn_tf_K_grad(batch_size, y_K_hat, y, W_K, W_MHN_tensor, W_out_tensor, x_reinst,
                            device=torch.device('cpu')):
    
    '''
    Computes gradient for W_K (through the MHN module) for a batch of data in the MHN-Transformer model

    batch_size      :   size of a gradient batch
    y_K_hat         :   output of Transformer model (batch_size x v_dim)
    y               :   ground truth output (batch_size x v_dim)
    W_K             :   key matrix (k_dim x input_dim)
    W_MHN_tensor    :   MHN weight matrices (batch_size x tf_dim x k_dim)
    W_out_tensor    :   output weight matrices (batch_size x v_dim x tf_dim)
    x_reinst        :   reinstated context vectors (batch_size x input_dim)
    device          :   device to perform the computations on
    '''

    K_grad = torch.zeros(W_MHN_tensor.shape[-1], x_reinst.shape[-1]).to(device)

    error = y_K_hat - y             # batch_size x v_dim

    # add up gradients from each context sequence in the batch
    for i in range(batch_size):
        W_MHN = W_MHN_tensor[i]     # tf_dim x k_dim
        W_out = W_out_tensor[i]     # v_dim x tf_dim
        tf_dim = W_MHN.shape[0]

        x_tilde = x_reinst[i]       # input_dim

        e = error[i]

        k = W_K @ x_tilde           # k_dim
        w = W_MHN @ k               # tf_dim
        sm = F.softmax(w, dim=0)
        # compute softmax Jacobian
        J_sm = torch.diag(sm) - torch.outer(sm, sm)

        first_term_K = 2 * e.reshape(1, -1) @ W_out @ J_sm
        first_term_K = first_term_K.reshape(-1)             # tf_dim

        K_tensor = W_MHN.unsqueeze(-1)                                              # tf_dim x k_dim x 1
        x_tilde_tensor = x_tilde.unsqueeze(0).repeat(tf_dim, 1).unsqueeze(1)        # tf_dim x 1 x input_dim
        big_tensor_for_K = torch.bmm(K_tensor, x_tilde_tensor)                      # tf_dim x k_dim x input_dim

        little_K_grad = torch.einsum('i,ijk->jk', first_term_K, big_tensor_for_K)
        K_grad += little_K_grad

    return K_grad

