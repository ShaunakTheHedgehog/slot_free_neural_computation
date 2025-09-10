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

  return attn_mask



# class implementing multi-head attention
# adapted from Effie Li and James McClelland's implementation found at 
# https://github.com/Effie-Li/transformer-structured-generalization-public

class Attention(nn.Module):

    '''
    Multi-head attention mechanism.
    '''

    def __init__(self, embed_dim, tf_dim, v_dim, n_heads=1, dropout=0., project_out=False, W_V_init=None):
        '''
        args
        ----
        embed_dim : int,
            the input dim for each item in sequence
        tf_dim : int
            the total dim of the transformer (should be divisible by n_heads)
        v_dim : int
            the dim of the value vectors (output vectors)
        n_heads : int
            number of attention heads
        dropout : float
            dropout prob applied to the attention weights
        project_out : bool
            whether to apply a final linear projection to the output of the attention layer
        W_V_init : torch.tensor
            if provided, initializes W_V to this value (should have shape (v_dim, embed_dim))
        '''
        super().__init__()

        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.attn_dim = tf_dim

        dim_head = tf_dim // n_heads
        assert dim_head * n_heads == tf_dim, "embed_dim must be divisible by num_heads"

        self.scale = (dim_head ** (-0.5))

        self.q_project = nn.Linear(embed_dim, tf_dim, bias=False)
        self.k_project = nn.Linear(embed_dim, tf_dim, bias=False)
        self.v_project = nn.Linear(embed_dim, v_dim, bias=False)

        # initialize Q, K, V weights
        with torch.no_grad():
            W_Q = torch.randn(tf_dim, embed_dim, requires_grad=True) * (0.25/tf_dim**0.5)
            W_K = torch.randn(tf_dim, embed_dim, requires_grad=True) * (0.25/tf_dim**0.5)
            W_V = torch.rand(v_dim, embed_dim, requires_grad=True) * 0.1
            self.q_project.weight.copy_(W_Q)
            self.k_project.weight.copy_(W_K)
            self.v_project.weight.copy_(W_V)

        if W_V_init is not None:
            with torch.no_grad():
                self.v_project.weight.copy_(W_V_init)

        self.attn_dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(tf_dim, embed_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, q, k, v, attn_mask=None, return_attn=False, attn_type='independent'):
        '''
        args
        ----
        q : torch.tensor
            shape `(batch, n_target_item, embed_dim)`       # for our case, n_target_item = n_source_item...
        k/v : torch.tensor
            shape `(batch, n_source_item, embed_dim)`
        attn_mask : torch.bool
            shape `(n_target_item, n_source_item)` or `(batch, n_target_item, n_source_item)`
            positions with ``True`` are allowed to attend while ``False`` are marked with -inf
        return_attn : bool
            whether to return attention weights
        attn_type : str
            type of attention mask to generate if attn_mask is None
            - 'causal' or 'independent'

        returns
        -------
        out : torch.tensor
            output, shape `(batch, n_target_item, embed_dim)`
        attn : torch.tensor
            attention weights (if return_attn), shape `(n_target_item, n_source_item)`
        '''

        batch_size, q_len, _ = q.shape
        _, k_len, _ = k.shape

        # check attention mask
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            assert attn_mask.shape == (q_len, k_len) or attn_mask.shape == (batch_size, q_len, k_len)
        else:
            attn_mask = generate_attn_mask(q_len, type=attn_type)

        # project q/k/v
        q = self.q_project(q) # (batch, n_items, tf_dim)
        k = self.k_project(k)
        v =  self.v_project(v)


        # (batch, n_items, n_heads x dim_head) -> (batch, n_heads, n_items, dim_head)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_heads), (q,k,v))

        attn = torch.matmul(q, k.transpose(-1,-2)) * self.scale # (batch, n_heads, n_target_items, n_source_items)
        if attn_mask is not None:
            if attn_mask.dim()==3:
                attn_mask = attn_mask.unsqueeze(1) # add an n_head dim for broadcast add
            attn_mask = attn_mask.to(attn.device)
            # mark -inf where mask==False
            ninf_mask = torch.zeros_like(attn_mask, dtype=q.dtype, device=attn.device)
            ninf_mask.masked_fill_(attn_mask==False, -1e9)#float('-inf'))
            attn += ninf_mask
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(self.attn_dropout(attn), v) # (batch, n_heads, n_target_items, n_source_items) x (batch, n_heads, n_source_item, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)') # (batch, n_target_items, n_heads x dim_head)
        out = self.to_out(out)      # (batch, n_target_items, embed_dim)
        return (out, attn) if return_attn else out
    



# generate a fixedreadout where the ith row of the readout corresponds to class i
def generate_fixed_readout(hidden_dim, output_dim, sigma=1.):
    readout = nn.Parameter(torch.randn(output_dim, hidden_dim) * sigma, requires_grad=False)
    return readout


# 3-layer multi-layer perceptron (MLP) with relu activation in the hidden layer
class MLP(nn.Module):

    '''
    3-layer MLP with relu activation in the hidden layer
    '''

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0., fixed_MLP_readout=False):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.layer2 = nn.Linear(hidden_dim, out_dim, bias=False)
        if fixed_MLP_readout:
             self.layer2.weight = generate_fixed_readout(hidden_dim, out_dim, sigma=1.)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, is_val=False):
        x = self.layer1(x)
        out = self.dropout(self.layer2(x))
        return out


# single transformer layer, consisting of a multi-head attention layer and an MLP
class TransformerLayer(nn.Module):

    '''
    single attention block, consisting of a self-attention layer and an MLP
    '''

    def __init__(self, embed_dim, tf_dim, n_heads, mlp_hid_dim, output_dim, dropout=0., fixed_MLP_readout=False):
        super().__init__()
        self.self_attn = Attention(embed_dim, tf_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(tf_dim, mlp_hid_dim, output_dim, dropout, fixed_MLP_readout)

    def forward(self, x, attn_mask=None, is_val=False):
        '''
        args
        ----
        x : tensor
            shape (batch, max_len, embed_dim)
        attn_mask : bool tensor
            - (n_items, n_items) or (batch, n_items, n_items)
            - mask for attention operation (e.g., causal future mask)
            - True will be attended, False will be masked with -inf
        '''
        # original = x.clone()    # (batch, n_target_dims, embed_dim)
        x = self.norm1(x + self.self_attn(x, x, x, attn_mask=attn_mask))
        x = self.mlp(x, is_val)     # (batch, n_target_dims, output_dim)

        return x


# simplified transformer layer with just a self-attention layer and no MLP
class SimplifiedTransformerLayer(nn.Module):

    '''
    single attention block, consisting of a self-attention layer only
    '''

    def __init__(self, embed_dim, tf_dim, n_heads, output_dim, W_V_init=None):
        super().__init__()
        self.self_attn = Attention(embed_dim, tf_dim, output_dim, n_heads, W_V_init=W_V_init)

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
        x = self.self_attn(x, x, x, attn_mask=None, attn_type=attn_type)

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


# squared error loss function for transformer model
def tf_mse_loss(model_outputs, target_labels):
    batch_size = len(target_labels)
    return torch.sum((model_outputs[:, -1, :] - target_labels)**2) # * (1./batch_size)


# train a transformer model on the case sequence task in batch mode
def train_tf_batchmode(model, full_seq_len, dataset_params, criterion,
                       regularizer=None, num_batches=20_000, batch_size=128, lr=1e-3,
                       toy_task_mode=False, reduced=False, freeze_K=False, freeze_Q=False, freeze_V=False, manual_grad_calc=False,
                       visualize_QKV_during=False, plot_mode=True, permutation_reduced=False, W_V_fixed=False,
                       full_key_covar=True, plot_freq=100, device=torch.device('cpu')):
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

    # setup progress bar
    pbar = trange(num_batches)
    pbar.set_description("---")

    dataset_name = dataset_params[0]
    assert dataset_name == 'case_sequence'
    assert len(dataset_params) == 2
    num_letters = dataset_params[1]

    batch_losses = []
    batch_accs = []

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

        if regularizer is not None:
            loss += regularizer(model)

        # evaluate the loss and accuracy on this batch
        loss = criterion(output, targets)
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
                                                         device=device)
        else:
            loss.backward()
            Q_grad, K_grad, V_grad = model.self_attn.q_project.weight.grad, model.self_attn.k_project.weight.grad, model.self_attn.v_project.weight.grad

        if W_V_fixed:
            V_grad = torch.zeros_like(W_V).to(device)

        # update Q, K, V weights
        update_tf_weights(model, Q_grad, K_grad, V_grad, lr, freeze_K=freeze_K, freeze_Q=freeze_Q, freeze_V=freeze_V)

        # get batch-averaged losses and add to train/val loss list
        curr_loss = loss.item() / batch_size
        batch_losses.append(curr_loss)
        batch_accs.append(train_acc.item())

        # periodically visualize learned Q, K, V weights and covariance matrices
        if visualize_QKV_during and (i % plot_freq == 0):
          _, _ = visualize_QKV_matrices(model, 'tf', label=f'Iteration {i}', W_V_lims=[-0.2, 1.2, 0.2], QK_lims=[-2, 5, 1])
          W_K = model.self_attn.k_project.weight.data
          _ = visualize_uppercase_lowercase_covariance(num_letters, W_K, label='', KK_lims=[-2, 4, 1], full=full_key_covar)

        pbar.set_description("Batch {:03} Train Loss {:.4f} Train Acc {:.4f}"\
                            .format(i+1, batch_losses[-1], batch_accs[-1]))

        pbar.update(1)

        # visualize learned Q, K, V weights and covariance matrices at the end of training
        if i == num_batches-1:
            wv, qk_submat = visualize_QKV_matrices(model, 'tf', label='', plot_mode=plot_mode, W_V_lims=[-0.2, 1.2, 0.2], QK_lims=[-2, 5, 1])
            W_K = model.self_attn.k_project.weight.data
            ul_cov = visualize_uppercase_lowercase_covariance(num_letters, W_K, label='', plot_mode=plot_mode, KK_lims=[-2, 4, 1], full=full_key_covar)


    return batch_losses, batch_accs, wv, ul_cov, qk_submat