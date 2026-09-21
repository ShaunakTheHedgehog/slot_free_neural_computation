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

# from einops import rearrange, repeat

from tqdm.auto import trange
from copy import deepcopy
import warnings
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import *
from dataset import generate_toy_case_sequence_dataset, generate_case_sequences, generate_permutation_reduced_dataset

# MODEL
#------------------------------------------------------------------------------

# generate a particular kind of attention mask for the self-attention layer ('causal' or 'independent')
# 'causal' means each item can only attend to previous items (including itself)
# 'independent' means each item attends to itself, and the last item attends to all previous items
def generate_attn_mask(seq_len, type='causal'):
  attn_mask = torch.ones(seq_len, seq_len)
  if type=='causal':
    attn_mask = torch.tril(torch.ones(seq_len, seq_len))
  elif type=='independent':
    # attn_mask = torch.zeros(seq_len, seq_len)
    attn_mask = torch.eye(seq_len, seq_len)
    attn_mask[-1, :-1] = torch.ones(seq_len-1)
    attn_mask[-1, -1] = 0

  return attn_mask.bool()



# class implementing multi-head attention
# adapted from Effie Li and James McClelland's implementation found at 
# https://github.com/Effie-Li/transformer-structured-generalization-public

class Attention(nn.Module):

    '''
    Single-head self-attention mechanism.
    '''

    def __init__(self, embed_dim, tf_dim, v_dim, dropout=0., W_V_init=None, beta=1.0):
        '''
        args
        ----
        embed_dim : int,
            the input dim for each item in sequence
        tf_dim : int
            the total dim of the transformer, which in this case is just the key/query vector dim
        v_dim : int
            the dim of the value vectors (output vectors)
        dropout : float
            dropout prob applied to the attention weights
        W_V_init : torch.tensor
            if provided, initializes W_V to this value (should have shape (v_dim, embed_dim))
        beta : float
            scaling factor for the attention weights
        '''
        super().__init__()

        # self.n_heads = n_heads
        # assert n_heads == 1, "Currently only supports single-head attention"

        self.embed_dim = embed_dim
        self.attn_dim = tf_dim
        self.v_dim = v_dim

        # dim_head = tf_dim // n_heads
        # assert dim_head * n_heads == tf_dim, "embed_dim must be divisible by num_heads"

        # self.scale = (dim_head ** (-0.5))
        self.beta = beta

        self.q_project = nn.Linear(embed_dim, tf_dim, bias=False)
        self.k_project = nn.Linear(embed_dim, tf_dim, bias=False)
        self.v_project = nn.Linear(embed_dim, v_dim, bias=False)

        # initialize Q, K, V weights
        with torch.no_grad():
            W_Q = torch.randn(tf_dim, embed_dim, requires_grad=True) * (1./tf_dim**0.5)
            W_K = torch.randn(tf_dim, embed_dim, requires_grad=True) * (1./tf_dim**0.5)
            W_V = torch.rand(v_dim, embed_dim, requires_grad=True) * 0.1
            self.q_project.weight.copy_(W_Q)
            self.k_project.weight.copy_(W_K)
            self.v_project.weight.copy_(W_V)

        if W_V_init is not None:
            with torch.no_grad():
                self.v_project.weight.copy_(W_V_init)

        self.attn_dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, return_attn=False, attn_type='independent'):
        '''
        args
        ----
        q : torch.tensor
            shape `(batch, seq_len, embed_dim)`      
        k : torch.tensor
            shape `(batch, seq_len, embed_dim)`
        v : torch.tensor
            shape `(batch, seq_len, v_dim)`
        attn_mask : torch.bool
            shape `(seq_len, seq_len)`, where
            positions with ``True`` are allowed to attend while ``False`` are marked with -1e9
        return_attn : bool
            whether to return attention weights
        attn_type : str
            type of attention mask to generate 
            - 'independent' or 'causal'

        returns
        -------
        out : torch.tensor
            output, shape `(batch, seq_len, embed_dim)`
        attn : torch.tensor
            attention weights (if return_attn), shape `(seq_len, seq_len)`
        '''

        batch_size, q_len, _ = q.shape      # here, q_len = k_len = seq_len
        _, k_len, _ = k.shape

        attn_mask = generate_attn_mask(q_len, type=attn_type)   # (seq_len, seq_len)

        # project q / k / v
        q = self.q_project(q)   # (batch, seq_len, tf_dim)
        k = self.k_project(k)   # (batch, seq_len, tf_dim)
        v =  self.v_project(v)  # (batch, seq_len, v_dim)

        # (batch, n_items, n_heads x dim_head) -> (batch, n_heads, n_items, dim_head)
        # q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_heads), (q,k,v))

        attn = torch.matmul(q, k.transpose(-1,-2)) * self.beta # (batch, seq_len, seq_len)

        # mask out attention weights for positions that are not allowed to attend
        attn_mask = attn_mask.to(attn.device)
        # mark -inf where mask==False
        ninf_mask = torch.zeros_like(attn_mask, dtype=q.dtype, device=attn.device)
        ninf_mask.masked_fill_(attn_mask==False, -1e9)  # float('-inf'))
        attn += ninf_mask

        # for each query vector, apply softmax over the dot products with all key vectors to get attention weights
        attn = F.softmax(attn, dim=-1)  

        out = torch.matmul(self.attn_dropout(attn), v)      # (batch, seq_len, seq_len) x (batch, seq_len, v_dim)
        # out = rearrange(out, 'b h n d -> b n (h d)') # (batch, n_target_items, n_heads x dim_head)
        # out = self.to_out(out)      # (batch, n_target_items, embed_dim)
        return (out, attn) if return_attn else out



# simplified transformer layer with just a self-attention layer and no MLP
class SimplifiedTransformerLayer(nn.Module):

    '''
    single attention block, consisting of a self-attention layer only
    '''

    def __init__(self, embed_dim, tf_dim, output_dim, W_V_init=None, beta=1.0):
        super().__init__()
        self.self_attn = Attention(embed_dim, tf_dim, output_dim, W_V_init=W_V_init, beta=beta)
        self.beta = beta

    def forward(self, x, attn_type='independent'):
        '''
        args
        ----
        x : tensor
            shape (batch, max_len, embed_dim)
        attn_type : str
            type of attention mask to generate 
            - 'causal' or 'independent'
        '''
        # original = x.clone()    # (batch, n_target_dims, embed_dim)
        x = self.self_attn(x, x, x, attn_type=attn_type)

        return x



# TRAINING BASELINE MINIMAL TRANSFORMER MODEL
#------------------------------------------------------------------------------

# update transformer weights with calculated gradients
def update_tf_weights(model, Q_grad, K_grad, V_grad, lr, 
                      freeze_K=False, freeze_Q=False, freeze_V=False):
    with torch.no_grad():
        if not freeze_Q:
            model.self_attn.q_project.weight.data = model.self_attn.q_project.weight.data - lr * Q_grad

        if not freeze_K:
            model.self_attn.k_project.weight.data = model.self_attn.k_project.weight.data - lr * K_grad

        if not freeze_V:
            model.self_attn.v_project.weight.data = model.self_attn.v_project.weight.data - lr * V_grad


# evaluate accuracy on a batch of training sequences
def get_val_acc(val_outputs, val_labels):
    logits = val_outputs[:, -1, :]
    acc = 1.*torch.sum((torch.argmax(logits, axis=-1) == torch.argmax(val_labels, axis=-1))) / len(logits)
    return acc


# train a transformer model on the case sequence task in batch mode
def train_tf_batchmode(model, full_seq_len, dataset_params, criterion,
                       regularizer=None, num_batches=5_000, batch_size=64, lr=5e-3,
                       toy_task_mode=False, reduced=False, freeze_K=False, freeze_Q=False, freeze_V=False, manual_grad_calc=False,
                       visualize_QKV_during=False, plot_mode=True, permutation_reduced=False, W_V_fixed=False,
                       full_key_covar=True, plot_freq=100, device=torch.device('cpu'), print_display=True):
    '''
    Key Arguments:
    model : nn.Module : transformer model to train
    full_seq_len : int : length of each input sequence (including query token)
    dataset_params : list : parameters for dataset generation
        - first element is the name of the dataset ('case_sequence')
        - second element is the number of distinct letters (e.g., if 4, letters are A, B, C, D)
    criterion : function : loss function to use (e.g. mse_loss)
    regularizer : function : if provided, a regularization function that takes the model as input and returns a scalar regularization loss
    num_batches : int : number of batches to train for
    batch_size : int : number of sequences per batch
    lr : float : learning rate

    Returns:
    batch_losses : list : list of training losses per batch
    batch_accs : list : list of training accuracies per batch
    wv : np.array : final learned W_V weight matrix
    ul_cov : np.array : final learned uppercase-lowercase covariance matrix (W_K^T W_K submatrix)
    qk_submat : np.array : final learned W_Q^T W_K submatrix
    '''

    if print_display:
        # setup progress bar
        pbar = trange(num_batches)
        pbar.set_description("---")

    dataset_name = dataset_params[0]
    assert dataset_name == 'case_sequence'
    assert len(dataset_params) == 2
    num_letters = dataset_params[1]

    batch_losses = []
    batch_accs = []
    wv, ul_cov, qk_submat = None, None, None 

    for i in range(num_batches):
        # first, generate a batch of training sequences
        inputs, targets = None, None
        if toy_task_mode:
            assert not permutation_reduced
            inputs, targets = generate_toy_case_sequence_dataset(reduced=reduced)
        elif permutation_reduced:
            inputs, targets = generate_permutation_reduced_dataset(num_letters)
        else:
            inputs, targets = generate_case_sequences(batch_size, full_seq_len-1, num_letters)

        inputs = inputs.to(device)
        targets = targets.to(device)

        output = model(inputs)

        # evaluate the loss and accuracy on this batch
        loss = criterion(output, targets)
        batch_loss = loss.item() / batch_size

        if regularizer is not None:
            loss += regularizer(model)

        train_acc = get_val_acc(output, targets)

        W_Q = model.self_attn.q_project.weight.data
        W_K = model.self_attn.k_project.weight.data
        W_V = model.self_attn.v_project.weight.data

        model.self_attn.q_project.weight.grad = None
        model.self_attn.k_project.weight.grad = None
        model.self_attn.v_project.weight.grad = None

        # update weights, manually or automatically
        if manual_grad_calc:
            Q_grad, K_grad, V_grad = calculate_QKV_grads(batch_size, output[:, -1, :], targets, W_Q, W_K, W_V, inputs,
                                                         device=device, beta=model.beta)
        else:
            loss.backward()
            Q_grad, K_grad, V_grad = model.self_attn.q_project.weight.grad, model.self_attn.k_project.weight.grad, model.self_attn.v_project.weight.grad

        if W_V_fixed:
            V_grad = torch.zeros_like(W_V).to(device)

        # update Q, K, V weights
        update_tf_weights(model, Q_grad, K_grad, V_grad, lr, freeze_K=freeze_K, freeze_Q=freeze_Q, freeze_V=freeze_V)

        # get batch-averaged losses and add to train/val loss list
        batch_losses.append(batch_loss)
        batch_accs.append(train_acc.item())

        # periodically visualize learned Q, K, V weights and covariance matrices
        if visualize_QKV_during and (i % plot_freq == 0):
          _, _ = visualize_QKV_matrices(model, 'tf', label=f'Iteration {i}') #, W_V_lims=[-0.2, 1.2, 0.2], QK_lims=[-2, 5, 1])
          W_K = model.self_attn.k_project.weight.data
          _ = visualize_uppercase_lowercase_covariance(num_letters, W_K, label='') #, KK_lims=[-2, 4, 1], full=full_key_covar)

        if print_display:
            pbar.set_description("Batch {:03} Train Loss {:.4f} Train Acc {:.4f}"\
                                .format(i+1, batch_losses[-1], batch_accs[-1]))

            pbar.update(1)

        # visualize learned Q, K, V weights and covariance matrices at the end of training
        if (i == num_batches-1):
            wv, qk_submat = visualize_QKV_matrices(model, 'tf', label='', plot_mode=plot_mode, W_V_lims=[-0.2, 1.2, 0.2], QK_lims=[-2, 5, 1])
            W_K = model.self_attn.k_project.weight.data
            ul_cov = visualize_uppercase_lowercase_covariance(num_letters, W_K, label='', plot_mode=plot_mode, KK_lims=[-2, 4, 1], full=full_key_covar)


    return batch_losses, batch_accs, wv, ul_cov, qk_submat


# verify that manually calculated gradients match up with automatic gradients (from PyTorch)
def compare_manual_vs_automatic_tf_training(model_params, full_seq_len, dataset_params, criterion,
                                            num_batches=5_000, batch_size=64, lr=1e-3, toy_task_mode=False,
                                            reduced=False, freeze_K=False, device=torch.device('cpu')):

    (embed_dim, tf_dim, output_dim) = model_params

    # create separate but identical copies, one to update automatically and one manually
    auto_model = SimplifiedTransformerLayer(embed_dim, tf_dim, output_dim).to(device)

    manual_model = SimplifiedTransformerLayer(embed_dim, tf_dim, output_dim).to(device)
    manual_model.load_state_dict(auto_model.state_dict())

    # setup progress bar
    pbar = trange(num_batches)
    pbar.set_description("---")

    dataset_name, num_letters = dataset_params

    # optimizer = optim.Adam(model.parameters(), lr=lr)

    auto_batch_losses = []
    auto_batch_accs = []
    manual_batch_losses = []
    manual_batch_accs = []

    for i in range(num_batches):
        if toy_task_mode:
            inputs, targets = generate_toy_case_sequence_dataset(reduced=reduced)
        else:
            inputs, targets = generate_case_sequences(batch_size, full_seq_len-1, num_letters)

        inputs = inputs.to(device)
        targets = targets.to(device)

        # automatic model update
        output = auto_model(inputs)
        auto_loss = criterion(output, targets)
        auto_train_acc = get_val_acc(output, targets)

        auto_WQ = auto_model.self_attn.q_project.weight.data
        auto_WK = auto_model.self_attn.k_project.weight.data
        auto_WV = auto_model.self_attn.v_project.weight.data

        curr_auto_loss = auto_loss.item() / batch_size
        auto_batch_losses.append(curr_auto_loss)
        auto_batch_accs.append(auto_train_acc.item())

        auto_model.self_attn.q_project.weight.grad = None
        auto_model.self_attn.k_project.weight.grad = None
        auto_model.self_attn.v_project.weight.grad = None
        auto_loss.backward()

        Q_grad, K_grad, V_grad = auto_model.self_attn.q_project.weight.grad, auto_model.self_attn.k_project.weight.grad, auto_model.self_attn.v_project.weight.grad

        update_tf_weights(auto_model, Q_grad, K_grad, V_grad, lr, freeze_K=freeze_K)

        # manual model update
        output = manual_model(inputs)
        manual_loss = criterion(output, targets)
        manual_train_acc = get_val_acc(output, targets)

        manual_batch_losses.append(manual_loss.item() / batch_size)
        manual_batch_accs.append(manual_train_acc.item())

        manual_WQ = manual_model.self_attn.q_project.weight.data
        manual_WK = manual_model.self_attn.k_project.weight.data
        manual_WV = manual_model.self_attn.v_project.weight.data

        # compare losses, accs, and weights
        assert np.isclose(curr_auto_loss, manual_loss.item() / batch_size, atol=1e-5), f"Losses differ at batch {i+1}: auto {curr_auto_loss}, manual {manual_loss.item() / batch_size}"
        assert np.isclose(auto_train_acc.item(), manual_train_acc.item(), atol=1e-5), f"Accs differ at batch {i+1}: auto {auto_train_acc.item()}, manual {manual_train_acc.item()}"
        assert torch.allclose(auto_WQ, manual_WQ, atol=1e-5), f"W_Q weights differ at batch {i+1}"
        assert torch.allclose(auto_WK, manual_WK, atol=1e-5), f"W_K weights differ at batch {i+1}"
        assert torch.allclose(auto_WV, manual_WV, atol=1e-5), f"W_V weights differ at batch {i+1}"

        Q_grad, K_grad, V_grad = calculate_QKV_grads(batch_size, output[:, -1, :], targets, manual_WQ, manual_WK, manual_WV, inputs, beta=manual_model.beta, device=device)

        update_tf_weights(manual_model, Q_grad, K_grad, V_grad, lr, freeze_K=freeze_K)

        pbar.set_description("Batch {:03} Auto Train Loss {:.4f} Auto Train Acc {:.4f} Manual Train Loss {:.4f} Manual Train Acc {:.4f}"\
                            .format(i+1, auto_loss.item(), auto_train_acc.item(), manual_loss.item(), manual_train_acc.item()))

        pbar.update(1)

    return auto_batch_losses, auto_batch_accs, manual_batch_losses, manual_batch_accs


if __name__ == "__main__":
    num_letters = 26
    tf_dim = 64
    C = 26
    auto_losses, auto_accs, manual_losses, manual_accs = compare_manual_vs_automatic_tf_training((3*num_letters, tf_dim, 2), C+1, ['case_sequence', num_letters], mse_loss,
                                                num_batches=5_000, batch_size=64, lr=1e-2, toy_task_mode=False,
                                                reduced=False, freeze_K=False, device=torch.device('cpu'))

    # plot
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(auto_losses, label='automatic')
    plt.plot(manual_losses, label='manual')
    plt.title('Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(auto_accs, label='automatic')
    plt.plot(manual_accs, label='manual')
    plt.title('Training Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
