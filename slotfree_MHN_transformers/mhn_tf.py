import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle as pkl

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
from baseline_tf import SimplifiedTransformerLayer, train_tf_batchmode


# class implementing a 1-winner MHN built to operate in parallel over a batch of inputs
class OneWinnerMHN(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, output_size, one_hot_input_size,
                 input_sparsity=None, softmax_beta=1., debug_mode=False, input_proj_strength=1.0, item_in_mhn=False):
        super().__init__()
        '''
        Arguments:
        ----
        batch_size: size of batches / number of MHNs running in parallel
        input_size: size of item + label vector
        hidden_size: MHN hidden size
        output_size: size of label (output) vector
        one_hot_input_size: size of one-hot encoded input vector (e.g. 3*num_letters for case sequence task)
        input_sparsity: fraction from 0 to 1 giving desired sparsity level of inputs (as 1s and 0s)
        softmax_beta: softmax inverse temperature parameter, used during the query step
        debug_mode: if True, sets up W_items to directly pass through input one-hot vectors to MHN hidden layer, 
                    enabling a new MHN hidden neuron to be recruited for each context item
        input_proj_strength: strength of identity projection from input to MHN hidden layer, if debug_mode is True
        '''
        num_letters = int(one_hot_input_size / 3.)
        num_context_tokens = 2 * num_letters

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.one_hot_input_size = one_hot_input_size
        self.beta = softmax_beta

        self.item_dim = input_size - output_size
        self.label_dim = output_size

        assert not item_in_mhn or debug_mode
        self.debug_mode = debug_mode
        self.item_in_mhn = item_in_mhn
        W_hi = torch.zeros(batch_size, hidden_size, self.item_dim)

        self.W_items = None
        # in case of debug mode, set up W_items to directly project input one-hot vectors to MHN hidden layer
        if debug_mode:
            if self.item_in_mhn:
                self.W_items = torch.zeros(batch_size, hidden_size, self.one_hot_input_size)
                self.W_items = self.W_items.uniform_()
            else:
                self.W_items = torch.zeros(hidden_size, self.one_hot_input_size)
                assert hidden_size >= num_context_tokens, "hidden_size must be >= number of context tokens in this case"
                self.W_items[:num_context_tokens, :num_context_tokens] = input_proj_strength * torch.eye(num_context_tokens)
                self.W_items = self.W_items.unsqueeze(0).repeat(batch_size, 1, 1)
            self.W_items = nn.Parameter(data=self.W_items)
        else:
            W_hi = W_hi.uniform_(0., 0.5)

        W_hi = nn.Parameter(data=W_hi)

        W_oh = torch.zeros(batch_size, self.label_dim, hidden_size)
        W_oh = W_oh.uniform_()
        W_oh = nn.Parameter(data=W_oh)

        self.W_hi = W_hi
        self.W_oh = W_oh


        W_reinst = torch.zeros(batch_size, hidden_size, one_hot_input_size)
        W_reinst = W_reinst.uniform_()
        self.W_reinst = W_reinst
        # self.W_reinst = self.W_reinst.to(device)

        self.input_sparsity = input_sparsity

    # updating MHN weights at a fixed timestep of processing (across a whole batch)
    # doing a one-winner-take-all weight update
    def __adjust_weights(self, z, x_curr, curr_context):
        # z is of shape batch_size x hidden_size
        # x_curr_item = x_curr[:, :self.item_dim]       # batch_size x item_dim
        # x_curr_label = x_curr[:, self.item_dim:]      # batch_size x label_dim
        # new_W_hi = self.W_hi.data + (torch.bmm(z.unsqueeze(-1), x_curr_item.unsqueeze(-2)) - self.W_hi.data * z.unsqueeze(-1))
        # new_W_oh = self.W_oh.data + (torch.bmm(x_curr_label.unsqueeze(-1), z.unsqueeze(-2)) - self.W_oh.data * z.unsqueeze(-2))

        # self.W_hi.data = new_W_hi
        # self.W_oh.data = new_W_oh

        winners = z.argmax(dim=1)                     # (batch_size,)
        batch_idx = torch.arange(z.shape[0], device=z.device)

        self.W_hi.data[batch_idx, winners, :] = x_curr[:, :self.item_dim]    # batch_size x item_dim
        self.W_oh.data[batch_idx, :, winners] = x_curr[:, self.item_dim:]    # batch_size x label_dim
        self.W_reinst[batch_idx, winners, :] = curr_context                  # batch_size x one_hot_input_size
        if self.debug_mode and self.item_in_mhn:
            self.W_items.data[batch_idx, winners, :] = curr_context     # batch_size x one_hot_input_size


    # loading 1-winner MHN buffer weights over each timestep in the context window
    def __context_forward_step(self, x_curr, curr_context):
        # x_curr has shape batch_size x (input_size + output_size) (taken at current timestep in sequence, before query)
        if self.debug_mode and self.item_in_mhn:
            hidden_logits = torch.bmm(self.W_items.data, curr_context.unsqueeze(-1))
        elif self.debug_mode and not self.item_in_mhn:
            hidden_logits = torch.bmm(self.W_hi.data, x_curr[:, :self.item_dim].unsqueeze(-1)) + torch.bmm(self.W_items.data, curr_context.unsqueeze(-1))  # batch_size x hidden_size x 1
        else:
            hidden_logits = torch.bmm(self.W_hi.data, x_curr[:, :self.item_dim].unsqueeze(-1))  # batch_size x hidden_size x 1
    
        z = sparsify_tensor(hidden_logits, 1, dim=1)   # use 1-winner-take-all; shape is batch_size x hidden_size x 1

        # winners = z.squeeze(-1).argmax(dim=1)
        # batch_idx = torch.arange(z.shape[0], device=z.device)
        # self.W_reinst[batch_idx, winners, :] = curr_context     # batch_size x one_hot_input_size

        # self.W_reinst = self.W_reinst + (torch.bmm(z, curr_context.unsqueeze(-2)) - self.W_reinst * z)

        self.__adjust_weights(z.squeeze(-1), x_curr, curr_context)

    # forward pass for query item
    def __query_forward_step(self, x_q):
        # x_curr has shape batch_size x (input_size + output_size) (taken at current timestep in sequence, before query)

        hidden_logits = torch.bmm(self.W_hi.data, x_q.unsqueeze(-1))  # batch_size x hidden_size x 1

        # z = sparsify_tensor(hidden_logits, 1, dim=1)   # use 1-winner-take-all; shape is batch_size x hidden_size x 1
        hidden_activations = F.softmax(self.beta * hidden_logits.squeeze(-1), dim=1)

        out = torch.bmm(self.W_oh.data, hidden_activations.unsqueeze(-1))
        out = out.squeeze(-1)                                      # shape is batch_size x output_size

        # use W_reinst to reactivate blended item vector
        reinst_context = torch.bmm(self.W_reinst.transpose(-2, -1), hidden_activations.unsqueeze(-1))  # batch_size x one_hot_input_size x 1
        reinst_context = reinst_context.squeeze(-1)         # batch_size x one_hot_input_size

        return out, reinst_context


    # batched online learning phase
    def forward(self, x_c, x_q, context_input):
        '''
        x_c            : embedded keys + values for items in context (batch_size x (seq_len - 1) x (input_size + output_size))
        x_q            : embedded representation for query item (batch_size x input_size)
        context_input  : one-hot vector representations of items in context (batch_size x (seq_len - 1) x one_hot_input_size)
        '''
        assert self.hidden_size >= x_c.shape[1], "MHN hidden_size must be >= number of items in context window"

        seq_len = x_c.shape[1] + 1
        for i in range(seq_len-1):
            x_curr = x_c[:, i, :]
            context_input_curr = context_input[:, i, :]     # batch_size x one_hot_input_size
            self.__context_forward_step(x_curr, curr_context=context_input_curr)

        out, reinst_context = self.__query_forward_step(x_q)

        return out, reinst_context


    # forward pass for reinstated item vector
    def reinst_context_forward(self, x_K_reinst):
        # x_K_reinst has shape batch_size x k_dim
        hidden_logits = torch.bmm(self.W_hi.data, x_K_reinst.unsqueeze(-1))  # batch_size x hidden_size x 1
        hidden_activations = F.softmax(self.beta * hidden_logits.squeeze(-1), dim=1)

        out = torch.bmm(self.W_oh.data, hidden_activations.unsqueeze(-1))
        out = out.squeeze(-1)

        return out


# MHN-based Transformer layer with 1-winner MHN as attention mechanism
class OneWinnerMHNLayer(nn.Module):

    def __init__(self, batch_size, input_dim, k_dim, v_dim, tf_dim, output_dim=None,
                 softmax_beta=1.0, project_out=False, debug_mode=False, item_in_mhn=False,
                 init_coeff=1.0, input_proj_strength=1.0, device=torch.device('cpu')):
        '''
        args
        ----
        batch_size : int,
            the number of sequences in each batch
        input_dim : int,
            the input dimension of the model
        k_dim : int,
            the input dim to the MHN
        v_dim : int,
            the output dim of the MHN
        tf_dim : int,
            the number of MHN hidden neurons
        output_dim : int or None,
            the output dimension of the model (if None, no output projection is done)
        softmax_beta : float,
            the inverse temperature parameter for the softmax function
        project_out : bool,
            whether to apply a final linear projection to the output of the attention layer
        debug_mode : bool,
            whether to enable "debug mode" in the MHN, where W_items is set to directly project 
            input one-hot vectors to MHN hidden layer
        item_in_mhn : bool,
            whether to include the input item in the MHN
        init_coeff : float,
            the initialization coefficient for the MHN weights
        input_proj_strength : float,
            the strength of the input projection
        device : torch.device
        '''

        super().__init__()

        self.input_dim = input_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.tf_dim = tf_dim
        self.batch_size = batch_size

        assert not item_in_mhn or debug_mode
        self.debug_mode = debug_mode
        self.item_in_mhn = item_in_mhn

        self.beta = softmax_beta

        self.input_proj_strength = input_proj_strength

        self.label_dim = v_dim
        self.item_dim = k_dim

        self.device = device

        # input is projected to item ('query') and label ('target value'), which are each embed_dim in length
        self.W_K = None
        self.W_V = None
        self.W_Q = None
        self.W_context_identity = nn.Identity(k_dim + v_dim)
        self.W_query_identity = nn.Identity(k_dim)

        self.W_V = nn.Linear(input_dim, v_dim, bias=False)
        self.W_K = nn.Linear(input_dim, k_dim, bias=False)
        self.W_Q = nn.Linear(input_dim, k_dim, bias=False)

        with torch.no_grad():
            q_weights = torch.randn(k_dim, input_dim) * (init_coeff/k_dim**0.5)
            k_weights = torch.randn(k_dim, input_dim) * (init_coeff/k_dim**0.5)
            v_weights = torch.rand(v_dim, input_dim) * 0.1
            
            self.W_Q.weight.copy_(q_weights)
            self.W_K.weight.copy_(k_weights)
            self.W_V.weight.copy_(v_weights)

        self.one_win_mhn = None

        self.W_project_out = nn.Linear(v_dim, output_dim, bias=False) if project_out else nn.Identity(v_dim)
        if project_out:
            assert output_dim is not None

        self.output_dim = output_dim

    # full forward pass
    def forward(self, x):
        context_input = x[:, :-1, :]
        # get embedded keys and values for context items
        x_K = self.W_K(context_input)
        x_V = self.W_V(context_input)
        x_context = torch.cat((x_K, x_V), dim=-1)       # batch_size x (seq_len - 1) x (k_dim + v_dim)

        x_Q = self.W_Q(x[:, -1, :])     # query embedding, has shape batch_size x k_dim

        # initialize new 1-winner MHN for loading up this batch of context items
        buffer_mhn = OneWinnerMHN(self.batch_size, self.item_dim + self.label_dim, self.tf_dim, self.v_dim, one_hot_input_size=self.input_dim,
                                  input_sparsity=None, softmax_beta=self.beta, debug_mode=self.debug_mode, input_proj_strength=self.input_proj_strength, item_in_mhn=self.item_in_mhn)

        self.one_win_mhn = buffer_mhn.to(self.device)
        self.one_win_mhn.W_reinst = self.one_win_mhn.W_reinst.to(self.device)

        # perform loading and query-based retrieval from the MHN
        mhn_out, reinst_context = self.one_win_mhn(x_context, x_Q, context_input)

        out = self.W_project_out(mhn_out)

        return out, reinst_context, x_Q

    # full forward pass for reinstantiated context input
    def reinst_context_forward(self, reinst_context):

        # full forward pass for reinstantiated context input
        # 'reinst_context' is batch_size x one_hot_input_size
        self.W_K.weight.data = self.W_K.weight.data.detach().requires_grad_(True)
        self.W_V.weight.data = self.W_V.weight.data.detach().requires_grad_(True)

        self.one_win_mhn.W_hi.data = self.one_win_mhn.W_hi.data.detach().requires_grad_(True)
        self.one_win_mhn.W_oh.data = self.one_win_mhn.W_oh.data.detach().requires_grad_(True)

        x_K = self.W_K(reinst_context)     # batch_size x k_dim
        x_V = self.W_V(reinst_context)     # batch_size x v_dim

        out = self.one_win_mhn.reinst_context_forward(x_K)     # batch_size x v_dim

        out = self.W_project_out(out)
        x_V_out = self.W_project_out(x_V)

        return out, x_V_out


# train any version of the MHN-based Transformer model (in batch mode)
def train_mhn_tf_model_batchmode(model, full_seq_len, dataset_params, criterion,
                                 num_batches=2_000, batch_size=64, lr=1e-3, toy_task_mode=False,
                                 reduced=False, freeze_K=False, freeze_Q=False, freeze_V=False,
                                 manual_grad_calc=False, visualize_QKV_during=False, K_lr=None, K_grad_type='through_MHN',
                                 plot_mode=True, permutation_reduced=False, plot_every=100, full_key_covar=True,
                                 WV_train_mode='via_reinstatement', device=torch.device('cpu'), print_display=True):
    
    '''
    Trains an MHN-based Transformer model in batch mode, with options to freeze Q, K, V 
    weights and to manually compute gradients.

    Key Arguments:
    model : OneWinnerMHNLayer : the MHN-based Transformer model to train
    full_seq_len : int : the full sequence length (including the query token)
    dataset_params : list : parameters for the dataset, e.g. ['case_sequence', num_letters]
    criterion : loss function, e.g. mse_loss
    num_batches : int : number of training batches
    batch_size : int : number of sequences per batch
    lr : float : learning rate for gradient descent
    K_grad_type : str : type of W_K gradient to use ('through_MHN', 'supervised', 'none')
    WV_train_mode : str : whether to train W_V 'via_reinstatement' or 'via_MHN_output'

    Returns:
    Q_losses : list : list of Q losses across each training batch
    K_losses : list : list of K losses across each training batch
    V_losses : list : list of V losses across each training batch
    accs : list : list of accuracies at each training batch
    wv : np.array : learned W_V weights at end of training
    ul_cov : np.array : learned uppercase-lowercase covariance matrix at end of training
    qk_submat : np.array : learned query-key covariance submatrix at end of training
    '''

    if print_display:
        # setup progress bar
        pbar = trange(num_batches)
        pbar.set_description("---")

    dataset_name = dataset_params[0]
    assert dataset_name == 'case_sequence'
    assert len(dataset_params) == 2
    assert WV_train_mode in ['via_reinstatement', 'via_MHN_output']
    assert manual_grad_calc or WV_train_mode == 'via_reinstatement', 'via_MHN_output has no autograd equivalent'

    num_letters = dataset_params[1]

    K_losses = []
    V_losses = []
    Q_losses = []

    batch_accs = []
    batch_losses = []

    auto_batch_accs = []
    auto_batch_losses = []

    wv, ul_cov, qk_submat = None, None, None 

    assert K_grad_type == 'through_MHN' or K_grad_type=='supervised' or K_grad_type == 'none'
    if freeze_K:
        assert K_grad_type == 'none'

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

        # forward pass through model + calculating accuracy at current batch
        Q_output, reinst_context, x_Q = model(inputs)
        train_acc = get_acc(Q_output, targets)
        batch_accs.append(train_acc.item())

        batch_loss = criterion(Q_output, targets)
        batch_losses.append(batch_loss.item() / batch_size)

        x_V_out = model.W_V(reinst_context)

        # capture current W_Q weights before they are updated
        W_Q_curr = model.W_Q.weight.data.clone()

        # calculate losses and update weights -- either manually or automatically
        manual_Q_grad, manual_K_grad, manual_V_grad = None, None, None
        if manual_grad_calc:
            manual_Q_grad, manual_V_grad = calculate_mhn_tf_QV_grads(batch_size, x_V_out, targets, model.W_Q.weight.data,
                                                                     model.one_win_mhn.W_hi.data, model.one_win_mhn.W_reinst,
                                                                     model.W_V.weight.data, inputs, reinst_context,
                                                                     device=device, WV_train_mode=WV_train_mode, MHN_out=Q_output, beta=model.beta)


        if not freeze_Q:
            # W_Q update

            # Q_loss = criterion(Q_output, targets)
            Q_loss = criterion(x_V_out, targets)
            Q_losses.append(Q_loss.item() / batch_size)

            if manual_grad_calc:
                model.W_Q.weight.data = model.W_Q.weight.data - lr * manual_Q_grad
            else:
                model.W_Q.weight.grad = None

                Q_loss.backward()
                with torch.no_grad():
                    model.W_Q.weight.data = model.W_Q.weight.data - lr * model.W_Q.weight.grad


        if not freeze_V:
            V_loss = criterion(x_V_out, targets)
            V_losses.append(V_loss.item() / batch_size)

            if manual_grad_calc:
                model.W_V.weight.data = model.W_V.weight.data - lr * manual_V_grad
            else:
                model.W_V.weight.grad = None
                reinst_context = reinst_context.detach().requires_grad_(True)
                x_V_out = model.W_V(reinst_context)

                V_loss = criterion(x_V_out, targets)
                V_loss.backward()
                with torch.no_grad():
                    model.W_V.weight.data = model.W_V.weight.data - lr * model.W_V.weight.grad


        if not freeze_K:
            assert K_grad_type in ['through_MHN', 'supervised']
            if K_lr is None:
              K_lr = lr
            reinst_context = reinst_context.detach().requires_grad_(True)
            K_loss = None 

            if K_grad_type == 'through_MHN':
                reinst_out_K, _ = model.reinst_context_forward(reinst_context)
                K_loss = criterion(reinst_out_K, targets)
            elif K_grad_type == 'supervised':
                # K_grad_type == 'supervised'
                K_loss = criterion(model.W_K(reinst_context), x_Q.detach())
            
            K_losses.append(K_loss.item() / batch_size)

            if manual_grad_calc:
                manual_K_grad = None

                if K_grad_type == 'through_MHN':
                    manual_K_grad = calculate_mhn_tf_K_grad(batch_size, reinst_out_K, targets, model.W_K.weight.data,
                                                    model.one_win_mhn.W_hi.data, model.one_win_mhn.W_oh.data, reinst_context, device=device, beta=model.beta)
                else: 
                    # K_grad_type == 'supervised'
                    manual_K_grad = simple_K_update(model.W_K.weight.data, reinst_context, W_Q_curr, inputs[:, -1, :], update_type='supervised', device=device)

                model.W_K.weight.data = model.W_K.weight.data - K_lr * manual_K_grad
            else:
                model.W_K.weight.grad = None
                K_loss.backward()
                with torch.no_grad():
                    model.W_K.weight.data = model.W_K.weight.data - K_lr * model.W_K.weight.grad

        # periodically visualize learned Q, K, V weights and covariance matrices
        if visualize_QKV_during and (i % plot_every == 0):
          _, _ = visualize_QKV_matrices(model, 'mhn_tf', label=f'(Iter {i})', W_V_lims=[0., 1., 0.2], QK_lims=[-2., 8., 2])
          _ = visualize_uppercase_lowercase_covariance(num_letters, model.W_K.weight.data, '', KK_lims=[-2., 8., 2], full=full_key_covar)

        curr_loss = None
        if len(Q_losses) == 0:
            curr_loss = V_losses[-1]
        else:
            curr_loss = Q_losses[-1]

        if print_display:
            pbar.set_description("Batch {:03} Train (Q) Loss {:.4f} Train Acc {:.4f}"\
                                .format(i+1, curr_loss, batch_accs[-1]))

            pbar.update(1)
        
        # visualize learned Q, K, V weights and covariance matrices at the end of training
        if (i==num_batches-1):
            wv, qk_submat = visualize_QKV_matrices(model, 'mhn_tf', label='', plot_mode=plot_mode, W_V_lims=[0., 1., 0.2], QK_lims=[-2., 8., 2])
            ul_cov = visualize_uppercase_lowercase_covariance(num_letters, model.W_K.weight.data, '', plot_mode=plot_mode, KK_lims=[-2., 8., 2], full=full_key_covar)

    return batch_losses, batch_accs, wv, ul_cov, qk_submat, Q_losses, K_losses, V_losses


# same as train_mhn_tf_model_batchmode, but with W_K fixed throughout training
def train_mhn_tf_model_batchmode_fixedK(model, full_seq_len, dataset_params, criterion,
                                        num_batches=20_000, batch_size=128, lr=1e-3, toy_task_mode=False,
                                        reduced=False, manual_grad_calc=False, visualize_QKV_during=False,
                                        plot_mode=True, permutation_reduced=False, full_key_covar=True,
                                        device=torch.device('cpu'), WV_train_mode='via_reinstatement', print_display=True):
    
    '''
    Trains an MHN-based Transformer model in batch mode with W_K fixed, with options to manually compute gradients.
    
    Key Arguments:
    model : OneWinnerMHNLayer : the MHN-based Transformer model to train
    full_seq_len : int : the full sequence length (including the query token)
    dataset_params : list : parameters for the dataset, e.g. ['case_sequence', num_letters]
    criterion : loss function, e.g. mse_loss
    num_batches : int : number of training batches
    batch_size : int : number of sequences per batch
    lr : float : learning rate for gradient descent
    WV_train_mode : str : whether to train W_V 'via_reinstatement' or 'via_MHN_output'

    Returns:
    batch_losses : list : list of losses across each training batch
    batch_accs : list : list of accuracies at each training batch
    wv : np.array : learned W_V weights at end of training
    ul_cov : np.array : learned uppercase-lowercase covariance matrix at end of training
    qk_submat : np.array : learned query-key covariance submatrix at end of training
    '''

    if print_display:
        # setup progress bar
        pbar = trange(num_batches)
        pbar.set_description("---")

    dataset_name = dataset_params[0]
    assert dataset_name == 'case_sequence'
    assert len(dataset_params) == 2
    num_letters = dataset_params[1]

    assert WV_train_mode in ['via_reinstatement', 'via_MHN_output']
    assert manual_grad_calc or WV_train_mode == 'via_reinstatement', 'via_MHN_output has no autograd equivalent'

    batch_losses = []
    batch_accs = []
    wv, ul_cov, qk_submat = None, None, None 

    for i in range(num_batches):
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

        Q_output, reinst_context, x_Q = model(inputs)
        train_acc = get_acc(Q_output, targets)
        batch_loss = criterion(Q_output, targets)

        # get batch avged losses and add to train/val loss list
        curr_loss = batch_loss.item() / batch_size
        batch_losses.append(curr_loss)
        batch_accs.append(train_acc.item())

        reinst_out = model.W_V(reinst_context)

        reinst_loss = criterion(reinst_out, targets)

        if manual_grad_calc:
            Q_grad, V_grad = calculate_mhn_tf_QV_grads(batch_size, reinst_out, targets, model.W_Q.weight.data, model.one_win_mhn.W_hi.data,
                                                model.one_win_mhn.W_reinst, model.W_V.weight.data, inputs, reinst_context, device=device,
                                                WV_train_mode=WV_train_mode, MHN_out=Q_output, beta=model.beta)
            model.W_Q.weight.data = model.W_Q.weight.data - lr * Q_grad
            model.W_V.weight.data = model.W_V.weight.data - lr * V_grad

        else:
            model.W_Q.weight.grad = None
            reinst_loss.backward()

            with torch.no_grad():
                model.W_Q.weight.data = model.W_Q.weight.data - lr * model.W_Q.weight.grad

            reinst_context = reinst_context.detach().requires_grad_(True)
            model.W_V.weight.grad = None

            reinst_out = model.W_V(reinst_context)
            V_loss = criterion(reinst_out, targets)
            V_loss.backward()

            with torch.no_grad():
                model.W_V.weight.data = model.W_V.weight.data - lr * model.W_V.weight.grad

        if visualize_QKV_during and (i % 500 == 0):
          _, _ = visualize_QKV_matrices(model, 'mhn_tf', label='')
          _ = visualize_uppercase_lowercase_covariance(num_letters, model.W_K.weight.data, '')

        if print_display:
            pbar.set_description("Batch {:03} Train Loss {:.4f} Train Acc {:.4f}"\
                                .format(i+1, curr_loss, batch_accs[-1]))

            pbar.update(1)

        if (i==num_batches-1):
            wv, qk_submat = visualize_QKV_matrices(model, 'mhn_tf', label='', plot_mode=plot_mode)
            ul_cov = visualize_uppercase_lowercase_covariance(num_letters, model.W_K.weight.data, '', plot_mode=plot_mode, full=full_key_covar)


    return batch_losses, batch_accs, wv, ul_cov, qk_submat


# run a sweep over multiple trials ('ntrials') of MHN-based Transformer model on case sequence task
def run_case_sequence_model_sweep(ntrials, model_type, num_letters, full_seq_len, k_dim, tf_dim,
                                  debug_mode, criterion, num_batches, batch_size, lr,
                                  K_lr=None, K_grad_type='through_MHN', final_window=1_000,
                                  device=torch.device('cpu'), manual_grad_calc=True, save_dir='',
                                  WV_train_mode='via_reinstatement', item_in_mhn=False, input_proj_strength=1.0,
                                  beta=1.0):

    '''
    Runs a sweep over multiple trials of an MHN-based Transformer model on the case sequence task.

    Key Arguments:
    ntrials : int : number of trials to run
    model_type : str : type of model to run ('tf', 'mhn_tf_fixed_WK', or 'mhn_tf')
    num_letters : int : number of letters in the case sequence task
    full_seq_len : int : full sequence length (including query token)
    k_dim : int : input dimension to the MHN
    tf_dim : int or None : number of MHN hidden neurons (in case model is not 'tf')
    debug_mode : bool : whether to enable "debug mode" in the MHN, thereby adding input projections
    criterion : loss function, e.g. mse_loss
    num_batches : int : number of training batches
    batch_size : int : number of context sequences per batch
    lr : float : learning rate for gradient descent 
    K_grad_type : str : type of W_K gradient to use ('through_MHN' (through MHN), 'supervised', or 'none')
    WV_train_mode : str : whether to train W_V 'via_reinstatement' or 'via_MHN_output'
    beta : float : inverse temperature multiplying the attention logits (baseline and MHN models alike)

    Returns:
    results_dict : dict : dictionary containing mean and all accuracies and losses, as well as weight and covariance stats
    all_accs : np.array : array of shape (ntrials, num_batches) containing accuracies for all trials and batches
    all_losses : np.array : array of shape (ntrials, num_batches) containing losses for all trials and batches
    covar_stats_dict : dict : dictionary containing mean covariance statistics
    mean_covar_stats_dict : dict : dictionary containing multi-trial-averaged mean covariance statistics
    '''

    assert model_type in ['tf', 'mhn_tf_fixed_WK', 'mhn_tf']
    assert K_grad_type in ['through_MHN', 'supervised', 'none']
    assert WV_train_mode in ['via_reinstatement', 'via_MHN_output']

    if save_dir != '':
        save_dir = save_dir + '/'

    input_dim = 3 * num_letters
    output_dim = 2
    dataset_params = ['case_sequence', num_letters]

    mean_accs = np.zeros(ntrials)
    mean_losses = np.zeros(ntrials)

    all_accs = np.zeros((ntrials, num_batches))
    all_losses = np.zeros((ntrials, num_batches))

    mean_W_V_diffs = np.zeros(ntrials)

    QK_l_diags_mean = np.zeros(ntrials)
    QK_l_diags_range = np.zeros(ntrials)
    QK_l_offdiags_mean = np.zeros(ntrials)
    QK_l_offdiags_range = np.zeros(ntrials)

    QK_u_diags_mean = np.zeros(ntrials)
    QK_u_diags_range = np.zeros(ntrials)
    QK_u_offdiags_mean = np.zeros(ntrials)
    QK_u_offdiags_range = np.zeros(ntrials)

    ul_diags_mean = np.zeros(ntrials)
    ul_diags_range = np.zeros(ntrials)
    ul_offdiags_mean = np.zeros(ntrials)
    ul_offdiags_range = np.zeros(ntrials)

    for i in range(ntrials):
        batch_losses, batch_accs, wv, ul_cov, qk_cov = None, None, None, None, None
        if model_type == 'tf':
            assert tf_dim is None
            assert K_grad_type == 'none'

            tf = SimplifiedTransformerLayer(input_dim, k_dim, output_dim, beta=beta).to(device)

            batch_losses, batch_accs, wv, ul_cov, qk_cov = train_tf_batchmode(tf, full_seq_len, dataset_params, criterion=criterion,
                                                                              num_batches=num_batches, batch_size=batch_size, lr=lr,
                                                                              manual_grad_calc=manual_grad_calc, plot_mode=False, device=device)

        elif model_type == 'mhn_tf_fixed_WK':
            assert tf_dim is not None
            assert K_grad_type == 'none'
            mhn_tf = OneWinnerMHNLayer(batch_size, input_dim, k_dim, output_dim, tf_dim, input_proj_strength=input_proj_strength, softmax_beta=beta,
                                       debug_mode=debug_mode, item_in_mhn=item_in_mhn, device=device).to(device)

            batch_losses, batch_accs, wv, ul_cov, qk_cov = train_mhn_tf_model_batchmode_fixedK(mhn_tf, full_seq_len, dataset_params, criterion=criterion,
                                                                                               num_batches=num_batches, batch_size=batch_size, lr=lr,
                                                                                               manual_grad_calc=manual_grad_calc, plot_mode=False, device=device,
                                                                                               WV_train_mode=WV_train_mode)
        else:
            assert model_type == 'mhn_tf'
            assert tf_dim is not None
            assert K_grad_type in ['through_MHN', 'supervised']
            mhn_tf = OneWinnerMHNLayer(batch_size, input_dim, k_dim, output_dim, tf_dim, input_proj_strength=input_proj_strength, softmax_beta=beta,
                                       debug_mode=debug_mode, item_in_mhn=item_in_mhn, device=device).to(device)

            batch_losses, batch_accs, wv, ul_cov, qk_cov, _, _, _ = train_mhn_tf_model_batchmode(mhn_tf, full_seq_len, dataset_params, criterion=criterion, num_batches=num_batches,
                                                                                              batch_size=batch_size, lr=lr, manual_grad_calc=manual_grad_calc, K_lr=K_lr, K_grad_type=K_grad_type,
                                                                                              plot_mode=False, device=device, WV_train_mode=WV_train_mode)



        batch_accs = np.array(batch_accs)
        batch_losses = np.array(batch_losses)

        mean_accs[i] = np.mean(batch_accs[-final_window:])
        mean_losses[i] = np.mean(batch_losses[-final_window:])

        all_accs[i] = batch_accs
        all_losses[i] = batch_losses

        mean_W_V_diffs[i] = get_W_V_stats(num_letters, wv)
        ul_diags_mean[i], ul_diags_range[i], ul_offdiags_mean[i], ul_offdiags_range[i] = get_upper_lower_covar_stats(num_letters, ul_cov[num_letters:, :num_letters])
        QK_l_diags_mean[i], QK_l_diags_range[i], QK_l_offdiags_mean[i], QK_l_offdiags_range[i], QK_u_diags_mean[i], QK_u_diags_range[i], QK_u_offdiags_mean[i], QK_u_offdiags_range[i] = get_QK_covar_stats(num_letters, qk_cov)

    mean_ul_covar_diffs = ul_diags_mean - ul_offdiags_mean
    mean_QK_l_covar_diffs = QK_l_diags_mean - QK_l_offdiags_mean
    mean_QK_u_covar_diffs = QK_u_diags_mean - QK_u_offdiags_mean

    # store results in dictionary and save to files
    mean_covar_stats_dict = {'ul_diags_mean_mean': ul_diags_mean.mean(), 'ul_diags_mean_range': ul_diags_range.mean(), 'ul_offdiags_mean_mean': ul_offdiags_mean.mean(), 'ul_offdiags_mean_range': ul_offdiags_range.mean(),
                            'QK_l_diags_mean_mean': QK_l_diags_mean.mean(), 'QK_l_diags_mean_range': QK_l_diags_range.mean(), 'QK_l_offdiags_mean_mean': QK_l_offdiags_mean.mean(), 'QK_l_offdiags_mean_range': QK_l_offdiags_range.mean(),
                            'QK_u_diags_mean_mean': QK_u_diags_mean.mean(), 'QK_u_diags_mean_range': QK_u_diags_range.mean(), 'QK_u_offdiags_mean_mean': QK_u_offdiags_mean.mean(), 'QK_u_offdiags_mean_range': QK_u_offdiags_range.mean()}

    covar_stats_dict = {'ul_diags_mean': ul_diags_mean, 'ul_diags_range': ul_diags_range, 'ul_offdiags_mean': ul_offdiags_mean, 'ul_offdiags_range': ul_offdiags_range,
                        'QK_l_diags_mean': QK_l_diags_mean, 'QK_l_diags_range': QK_l_diags_range, 'QK_l_offdiags_mean': QK_l_offdiags_mean, 'QK_l_offdiags_range': QK_l_offdiags_range,
                        'QK_u_diags_mean': QK_u_diags_mean, 'QK_u_diags_range': QK_u_diags_range, 'QK_u_offdiags_mean': QK_u_offdiags_mean, 'QK_u_offdiags_range': QK_u_offdiags_range}

    results_dict = {'mean_final_accs': mean_accs, 'mean_final_losses': mean_losses,
                   'mean_W_V_diffs': mean_W_V_diffs, 'mean_ul_covar_diffs': mean_ul_covar_diffs,
                   'mean_QK_l_covar_diffs': mean_QK_l_covar_diffs, 'mean_QK_u_covar_diffs': mean_QK_u_covar_diffs}

    # concatenate all three dictionaries into one for saving
    full_results_dict = {**results_dict, **covar_stats_dict, **mean_covar_stats_dict, 'losses': all_losses, 'accs': all_accs}

    # save as pkl file to directory
    with open(f'{save_dir}full_results_{model_type}_ntrials{ntrials}_L{num_letters}_C{full_seq_len-1}_kdim{k_dim}_tfdim{tf_dim}_debugmode{debug_mode}_Kgrad_{K_grad_type}_iteminmhn_{item_in_mhn}.pkl', 'wb') as f:
        pkl.dump(full_results_dict, f)

    print(f'Results Dictionary: {results_dict}')
    print(f'\nCovariance Matrices Stats: {covar_stats_dict}')
    print(f'\nMean Covariance Matrices Stats: {mean_covar_stats_dict}')

    print(f'\nMedian Accuracy: {np.median(all_accs, 0)}')
    print(f'\nMedian Loss: {np.median(all_losses, 0)}')

    return full_results_dict


def compare_manual_vs_auto(model_params, full_seq_len, dataset_params, criterion,
                                 num_batches=2_000, batch_size=64, lr=1e-3, 
                                 freeze_K=False, freeze_Q=False, freeze_V=False,
                                 K_lr=None, K_grad_type='through_MHN', add_proj=False, item_in_mhn=False,
                                 WV_train_mode='via_reinstatement', device=torch.device('cpu')):
    
    (batch_size, input_dim, k_dim, v_dim, tf_dim) = model_params
    
    # create separate but identical copies, one to update automatically and one manually
    auto_model = OneWinnerMHNLayer(batch_size, input_dim, k_dim, v_dim, tf_dim, debug_mode=add_proj, item_in_mhn=item_in_mhn).to(device)

    manual_model = deepcopy(auto_model)

    if freeze_K:
        auto_model_fixedK = deepcopy(auto_model)
        manual_model_fixedK = deepcopy(manual_model)
    # OneWinnerMHNLayer(batch_size, input_dim, k_dim, v_dim, tf_dim, debug_mode=add_proj, item_in_mhn=item_in_mhn).to(device)
    # manual_model.load_state_dict(auto_model.state_dict())

    # setup progress bar
    pbar = trange(num_batches)
    pbar.set_description("---")

    all_auto_losses = []
    all_auto_accs = []

    all_manual_losses = []
    all_manual_accs = []

    # now, compare the models' training performance (weights, loss, acc) step by step
    for i in range(num_batches):
        seed = 1234 + i

        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        auto_WQ = auto_model.W_Q.weight.data.clone()
        auto_WK = auto_model.W_K.weight.data.clone()
        auto_WV = auto_model.W_V.weight.data.clone()

        # first run train_tf_batchmode for one step on each model
        auto_losses, auto_accs, auto_wv, auto_ul_cov, auto_qk_submat, *_ = train_mhn_tf_model_batchmode(auto_model, full_seq_len, dataset_params, criterion,
                                                                                          num_batches=1, batch_size=batch_size, lr=lr,
                                                                                          freeze_K=freeze_K, freeze_Q=freeze_Q, freeze_V=freeze_V,
                                                                                          manual_grad_calc=False, visualize_QKV_during=False,
                                                                                          K_lr=K_lr, K_grad_type=K_grad_type, device=device,
                                                                                          WV_train_mode=WV_train_mode, plot_mode=False, print_display=False)

        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        manual_WQ = manual_model.W_Q.weight.data.clone()
        manual_WK = manual_model.W_K.weight.data.clone()
        manual_WV = manual_model.W_V.weight.data.clone()

        manual_losses, manual_accs, manual_wv, manual_ul_cov, manual_qk_submat, *_ = train_mhn_tf_model_batchmode(manual_model, full_seq_len, dataset_params, criterion,
                                                                                          num_batches=1, batch_size=batch_size, lr=lr,
                                                                                          freeze_K=freeze_K, freeze_Q=freeze_Q, freeze_V=freeze_V,
                                                                                          manual_grad_calc=True, visualize_QKV_during=False,
                                                                                          K_lr=K_lr, K_grad_type=K_grad_type, device=device,
                                                                                          WV_train_mode=WV_train_mode, plot_mode=False, print_display=False)

        if freeze_K:
            # in this case, check that train_tf_batchmode_fixedK gives the same results as train_tf_batchmode with freeze_K=True
            auto_WQ_fixedK = auto_model_fixedK.W_Q.weight.data.clone()
            auto_WK_fixedK = auto_model_fixedK.W_K.weight.data.clone()
            auto_WV_fixedK = auto_model_fixedK.W_V.weight.data.clone()

            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            auto_losses_fixedK, auto_accs_fixedK, auto_wv_fixedK, auto_ul_cov_fixedK, auto_qk_submat_fixedK = train_mhn_tf_model_batchmode_fixedK(auto_model_fixedK, full_seq_len, dataset_params, criterion,
                                                                                          num_batches=1, batch_size=batch_size, lr=lr,
                                                                                          manual_grad_calc=False, visualize_QKV_during=False,
                                                                                          device=device,
                                                                                          WV_train_mode=WV_train_mode, plot_mode=False, print_display=False)      

            manual_WQ_fixedK = manual_model_fixedK.W_Q.weight.data.clone()
            manual_WK_fixedK = manual_model_fixedK.W_K.weight.data.clone()
            manual_WV_fixedK = manual_model_fixedK.W_V.weight.data.clone()

            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            manual_losses_fixedK, manual_accs_fixedK, manual_wv_fixedK, manual_ul_cov_fixedK, manual_qk_submat_fixedK = train_mhn_tf_model_batchmode_fixedK(manual_model_fixedK, full_seq_len, dataset_params, criterion,
                                                                                          num_batches=1, batch_size=batch_size, lr=lr,
                                                                                          manual_grad_calc=True, visualize_QKV_during=False,
                                                                                          device=device,
                                                                                          WV_train_mode=WV_train_mode, plot_mode=False, print_display=False)                                                                       

        # compare the losses, accuracies, and weights of the two models
        tol = 1e-5
        assert np.isclose(auto_losses[-1], manual_losses[-1], atol=tol), f"Losses differ at batch {i}: auto {auto_losses[-1]}, manual {manual_losses[-1]}"
        assert np.isclose(auto_accs[-1], manual_accs[-1], atol=tol), f"Accuracies differ at batch {i}: auto {auto_accs[-1]}, manual {manual_accs[-1]}"
        # assert np.allclose(auto_wv, manual_wv, atol=tol), f"W_V weights differ at batch {i}"
        # assert np.allclose(auto_ul_cov, manual_ul_cov, atol=tol), f"Upper-lower covariance matrices differ at batch {i}"
        # assert np.allclose(auto_qk_submat, manual_qk_submat, atol=tol), f"Query-key covariance submatrices differ at batch {i}"
        assert np.allclose(auto_model.W_Q.weight.data, manual_model.W_Q.weight.data, atol=tol), f"W_Q weights differ at batch {i}"
        assert np.allclose(auto_model.W_K.weight.data, manual_model.W_K.weight.data, atol=tol), f"W_K weights differ at batch {i}"
        assert np.allclose(auto_model.W_V.weight.data, manual_model.W_V.weight.data, atol=tol), f"W_V weights differ at batch {i}"

        if freeze_K:
            assert np.isclose(auto_losses_fixedK[-1], manual_losses_fixedK[-1], atol=tol), f"Losses differ at batch {i} (fixed K): auto {auto_losses_fixedK[-1]}, manual {manual_losses_fixedK[-1]}"
            assert np.isclose(auto_accs_fixedK[-1], manual_accs_fixedK[-1], atol=tol), f"Accuracies differ at batch {i} (fixed K): auto {auto_accs_fixedK[-1]}, manual {manual_accs_fixedK[-1]}"
            # assert np.allclose(auto_wv_fixedK, manual_wv_fixedK, atol=tol), f"W_V weights differ at batch {i} (fixed K)"
            # assert np.allclose(auto_ul_cov_fixedK, manual_ul_cov_fixedK, atol=tol), f"Upper-lower covariance matrices differ at batch {i} (fixed K)"
            # assert np.allclose(auto_qk_submat_fixedK, manual_qk_submat_fixedK, atol=tol), f"Query-key covariance submatrices differ at batch {i} (fixed K)"
            assert np.allclose(auto_model_fixedK.W_Q.weight.data, manual_model_fixedK.W_Q.weight.data, atol=tol), f"W_Q weights differ at batch {i} (fixed K)"
            assert np.allclose(auto_model_fixedK.W_K.weight.data, manual_model_fixedK.W_K.weight.data, atol=tol), f"W_K weights differ at batch {i} (fixed K)"
            assert np.allclose(auto_model_fixedK.W_V.weight.data, manual_model_fixedK.W_V.weight.data, atol=tol), f"W_V weights differ at batch {i} (fixed K)"

            # also check that the fixed K versions match the non-fixed K versions when freeze_K=True
            assert np.isclose(auto_losses_fixedK[-1], auto_losses[-1], atol=tol), f"Losses differ at batch {i} (fixed K vs non-fixed K): auto {auto_losses_fixedK[-1]}, auto {auto_losses[-1]}"
            assert np.isclose(manual_losses_fixedK[-1], manual_losses[-1], atol=tol), f"Losses differ at batch {i} (fixed K vs non-fixed K): manual {manual_losses_fixedK[-1]}, manual {manual_losses[-1]}"
            assert np.isclose(auto_accs_fixedK[-1], auto_accs[-1], atol=tol), f"Accuracies differ at batch {i} (fixed K vs non-fixed K): auto {auto_accs_fixedK[-1]}, auto {auto_accs[-1]}"
            assert np.isclose(manual_accs_fixedK[-1], manual_accs[-1], atol=tol), f"Accuracies differ at batch {i} (fixed K vs non-fixed K): manual {manual_accs_fixedK[-1]}, manual {manual_accs[-1]}"
            # assert np.allclose(auto_wv_fixedK, auto_wv, atol=tol), f"W_V weights differ at batch {i} (fixed K vs non-fixed K)"
            # assert np.allclose(manual_wv_fixedK, manual_wv, atol=tol), f"W_V weights differ at batch {i} (fixed K vs non-fixed K)"
            # assert np.allclose(auto_ul_cov_fixedK, auto_ul_cov, atol=tol), f"Upper-lower covariance matrices differ at batch {i} (fixed K vs non-fixed K)"
            # assert np.allclose(manual_ul_cov_fixedK, manual_ul_cov, atol=tol), f"Upper-lower covariance matrices differ at batch {i} (fixed K vs non-fixed K)"
            # assert np.allclose(auto_qk_submat_fixedK, auto_qk_submat, atol=tol), f"Query-key covariance submatrices differ at batch {i} (fixed K vs non-fixed K)"
            # assert np.allclose(manual_qk_submat_fixedK, manual_qk_submat, atol=tol), f"Query-key covariance submatrices differ at batch {i} (fixed K vs non-fixed K)"
            assert np.allclose(auto_WQ_fixedK, auto_WQ, atol=tol), f"W_Q weights differ at batch {i} (fixed K vs non-fixed K)"
            assert np.allclose(manual_WQ_fixedK, manual_WQ, atol=tol), f"W_Q weights differ at batch {i} (fixed K vs non-fixed K)"
            assert np.allclose(auto_WK_fixedK, auto_WK, atol=tol), f"W_K weights differ at batch {i} (fixed K vs non-fixed K)"
            assert np.allclose(manual_WK_fixedK, manual_WK, atol=tol), f"W_K weights differ at batch {i} (fixed K vs non-fixed K)"
            assert np.allclose(auto_WV_fixedK, auto_WV, atol=tol), f"W_V weights differ at batch {i} (fixed K vs non-fixed K)"
            assert np.allclose(manual_WV_fixedK, manual_WV, atol=tol), f"W_V weights differ at batch {i} (fixed K vs non-fixed K)"

        all_auto_losses.append(auto_losses[-1])
        all_auto_accs.append(auto_accs[-1])
        all_manual_losses.append(manual_losses[-1])
        all_manual_accs.append(manual_accs[-1])

        # update display
        pbar.set_description("Batch {:03} Auto Train Loss {:.4f} Auto Train Acc {:.4f} Manual Train Loss {:.4f} Manual Train Acc {:.4f}"\
                            .format(i+1, all_auto_losses[-1], all_auto_accs[-1], all_manual_losses[-1], all_manual_accs[-1]))

        pbar.update(1)

    return all_auto_losses, all_auto_accs, all_manual_losses, all_manual_accs


if __name__ == "__main__":
    B = 64
    L = 4
    C = 4
    lr = 1e-3
    K_lr = None
    model_params = (B, 3*L, 32, 2, 8)  # batch_size, input_dim, k_dim, v_dim, tf_dim
    dataset_params = ['case_sequence', L]
    criterion = mse_loss 
    K_grad_type = 'none'
    freeze_K = True
    add_proj = False
    item_in_mhn = False

    auto_losses, auto_accs, manual_losses, manual_accs = compare_manual_vs_auto(model_params, C+1, dataset_params, criterion,
                                    num_batches=5_000, batch_size=B, lr=lr, 
                                    freeze_K=freeze_K, freeze_Q=False, freeze_V=False,
                                    K_lr=K_lr, K_grad_type=K_grad_type, add_proj=add_proj, item_in_mhn=item_in_mhn,
                                    WV_train_mode='via_reinstatement', device=torch.device('cpu'))


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