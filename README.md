# Neural Computation Without Slots
This repository contains the code used to run all simulations and generate all figures for the paper "Neural Computation Without Slots: Steps Towards Biologically Plausible Memory and Attention in Natural and Artificial Intelligence." Inspired by the Modern Hopfield Network (MHN) framework from Krotov & Hopfield (2020), this project consists of 2 core components:

1. Designing a slot-free sparse yet *distributed* connectionist memory model, enabling memories to be represented by a distributed set of memory neurons, with each memory neuron possibly becoming activated across multiple different memories.

2. Designing and training slot-free, connection weight-based instantiations of a "minimal" Transformer architecture in a way that avoids needing to actively maintain arbitrararily long temporal sequences of prior states in order to perform credit assignment.

Ultimately, both of these components constitute steps towards understanding how *biologically plausible mechanisms* can support computations that have enabled AI systems to capture human-like abilities that no prior models have been able to achieve.


# A Slot-free, Sparse, Distributed Memory Model: The K-winner Modern Hopfield Network

This repository contains code for the *K-winner Modern Hopfield Network* (K-winner MHN), a distributed associative memory model developed as part of the broader notion of *slot-free neural computation*.  

The K-winner MHN extends the Modern Hopfield Network (MHN) framework by introducing a **k-winner-take-all rule** at the hidden layer, thereby enforcing sparse yet distributed neural representations, akin to real biological neural representations of memories. The K-winner MHN framework also allows for sparse network connnectivity and graded weight updates, altogether affording greater retention of older memories.

The functions provided here enable replication of the key retrieval and sensitivity analyses in the accompanying research manuscript. The main functions documented below can be found in the code file `kwinner_samhn_comparison.py`, and this code relies on the accompanying files `utils.py` and `kwinnernet.py`.

---

## Core Experimental Functions

| Function | Description | Used for |
|-----------|--------------|----------|
| `get_retrieval_probability_comparison(...)` | Compares retrieval accuracy curves across memory age between two networks (e.g. K-winner vs 1-winner). | Figures 4A, S2 (retrieval accuracy curves) |
| `run_comparison_test(...)` | Compares discrimination sensitivity \( d' \) and raw difference (R.D.) curves between two networks. | Figures 4B, 5, S1, S2, S3 |
| `get_match_probabilities(...)` | Internal helper called by the above functions; computes trial-averaged retrieval performance. | (not used directly) |
| `plot_acc_curves(...)` | Plots retrieval accuracy curves (real vs pseudo patterns). | Figures 4A, S6 |
| `plot_data(...)` | Plots \( d' \) or R.D. curves across memory age. | Figures 4B, 5, S1–S3 |

---

## Dependencies

Ensure the following dependencies are available:

- numpy  
- matplotlib  
- scipy  
- scikit-learn  
- pickle  


## Parameter Conventions

Each model configuration (“runset”) is represented as a tuple:

```bash
runset = (n_i, n_h, s, f, k, epsilon)
```

where:
- n_i – number of input units
- n_h – number of hidden units
- s – input sparsity (active bits per pattern)
- f – fan-in scaling factor
- k – number of active (winning) hidden units
- epsilon – hidden-to-output scaling parameter

---

## Reproducing Main Figures

Below are template scripts for regenerating the figures from the paper.
All calls can be executed from within a Python environment after importing this module.


### Retrieval Accuracy Across Memory Age

The template code below shows how to generate a plot comparing the retrieval accuracy (across memory ages) for a K-winner MHN and a parameter-matched Original MHN, using 100% / 50% cues. It can be used for both unstructured patterns and hierarchical, tree-generated patterns. This template code can be used to generate Fig. 4A and SI Appendix Fig. S2.

```bash
from kwinner_samhn_comparison import get_retrieval_probability_comparison, plot_acc_curves

# for conserving total parameter count, we require that n_h * f = n_h_MHN
runset1 = (n_i, n_h, s, f, k, epsilon)      # K-winner MHN
runset2 = (n_i, n_h_MHN, s, 1., 1, 1.)      # 1-winner (original) MHN

results = get_retrieval_probability_comparison(
    runset1, runset2,
    num_trials=100,
    num_mems=2_000,
    cue_level=1.0,          # or 0.5 for partial cues
    filename='fig_4A_retrieval_accs',
    save_data=True,
    data_clustered=False,    # set to True if using structured, hierarchical patterns as input ("tree data")
    num_flips=None           # if using tree data, specify the bit flips parameter b
    uniform_baseline=False   # if using tree data, specify whether an additional baseline for *unstructured* patterns should be computed 
)

plot_acc_curves(
    data=[(results['model1_out'], results['model1_pseudo_out'], results['model1_unif_pseudo_out']),
          (results['model2_out'], results['model2_pseudo_out'], results['model2_unif_pseudo_out'])],
    runsets=(runset1, runset2),
    max_age=1_000,    # maximum age of memory to show in figure
    plot_title='Fig 4A Retrieval Accuracy Curves',
    plot_xlabel='Memory Age',
    plot_ylabel='Retrieval Accuracy',
    figsize=(8, 5),
    uniform_baseline=False,    # show uniform, unstructured pseudo-pattern baseline in case training on structured, hierarchical data
    legend=False
)

```


### Retrieval Sensitivity (d') Comparison

The template code below shows how to generate a figure comparing the retrieval sensitivity across memory age, as measured by d', for a K-winner MHN and a parameter-matched Original MHN, using 100% / 50% cues. It can be used for both unstructured patterns and hierarchical, tree-generated patterns. This template code can be used to generate Fig. 4B, Fig. 5 and SI Appendix Figs. S2 and S3.

```bash

from kwinner_samhn_comparison import run_comparison_test, plot_data

# for conserving total parameter count, we require that n_h * f = n_h_MHN
runset1 = (n_i, n_h, s, f, k, epsilon)      # K-winner MHN
runset2 = (n_i, n_h_MHN, s, 1., 1, 1.)      # 1-winner (original) MHN

results_dict = run_comparison_test(
    runset1, runset2,
    num_mems=2_000,
    num_samples=10,
    num_runs_per_sample=20,
    cue_level=1.0,           # or 0.5 for 50% cues
    data_clustered=False,    # specify as True if using structured, tree-generated patterns
    num_flips=None,          # if using tree data, specify bit flips parameter b
    save_data=True,
    filename='fig_4B_dprimes'
)

kwinner_data = results_dict['kwinner_dprimes']
mhn_data = results_dict['mhn_dprimes']

plot_data(
    kwinner_data,
    mhn_data,
    plot_title='Fig 4B d' Curves',
    max_age=1_000,
    plot_xlabel='100% Cues',    # adjust accordingly if plotting for 50% cues
    plot_ylabel='d\'',
    plot_rel_advantages=True
)

```

### Comparison of Raw Difference (R.D.) Curves

The template code below shows how to generate a figure comparing the raw difference (i.e. the difference between retrieval accuracy and baseline pseudo-pattern retrieval accuracy) across memory age, for a K-winner MHN and a parameter-matched Original MHN, using 100% / 50% cues. It can be computed for both unstructured patterns and hierarchical, tree-generated patterns. This template code can be used to generate SI Appendix Fig. S1.

```bash

from kwinner_samhn_comparison import run_comparison_test, plot_data

# for conserving total parameter count, we require that n_h * f = n_h_MHN
kwinner_runset = (n_i, n_h, s, f, k, epsilon)      # K-winner MHN
mhn_runset = (n_i, n_h_MHN, s, 1., 1, 1.)      # 1-winner (original) MHN

results_dict = run_comparison_test(
    kwinner_runset, mhn_runset,
    num_mems=2_000,
    num_samples=10,
    num_runs_per_sample=20,
    cue_level=1.0,        # or 0.5, if using 50% cues
    data_clustered=False,    # specify as True if using structured, tree-generated patterns
    num_flips=None,          # if using tree data, specify bit flips parameter b
    save_data=True,
    filename='fig_S1_rawdiffs'
)

kwinner_data = results_dict['kwinner_rawdiffs']
mhn_data = results_dict['mhn_rawdiffs']

plot_data(
    kwinner_data,
    mhn_data,
    plot_title='Fig S1 Raw Difference Curves',
    max_age=1_000,
    plot_xlabel='100% Cues',      # adjust accordingly if plotting for 50% cues
    plot_ylabel='Raw Differences',
    plot_rel_advantages=True,
    plot_mode='log_y',            # set to 'log_y' if using log scale, else set to None
    with_regression=True,         # plot regressed exponential decay curves to observed raw difference curves
    regression_idxs=200,          # specify which memory ages are used for regression (e.g. ages 1-200)
    mhn_runset=mhn_runset,        # specify the runset used for the MHN (so that the MHN theory/analytical R.D. curve can also be calculated)
    cue_level=1.0                 # specify cue level used
)

```

## Reproducibility Notes

All results are averaged over multiple independent trials for statistical robustness. The number of runs (num_trials, num_samples, num_runs_per_sample) can be adjusted to trade off accuracy and compute time. 

---


# A Slot-free, Connection Weight-based Implementation of the Transformer

This repository contains code for our *MHN-Transformer* models, which constitute biologically plausible variants of a minimal Transformer architecture and form part of the broader notion of *slot-free neural computation*.  

Our MHN-Transformer models utilize the Modern Hopfield Network (MHN) framework by incorporating a *hetero-associative MHN* "memory module" that reproduces the standard transformer self-attention mechanism. We also augment this MHN architecture with "item reinstatement" weights that allows for the reinstatement of best-matching context items when processing a query element. In our paper, we describe several variants of our connection weight-based transformer system that enables the learning of weights that encode representations of past inputs, in order to enhance their utilization for processing inputs at later times.

The functions provided here enable replication of the key retrieval and sensitivity analyses in the accompanying research manuscript. The main functions documented below can be found in the code files `mhn_tf.py` (for our MHN-based Transformer models), `baseline_tf.py` (for the baseline minimal Transformer architecture), and `dataset.py` (for implementing a canonical in-context learning task), and this code relies on the accompanying file `utils.py`.

---


### Core Classes and Functions

#### `OneWinnerMHN`
A batch-parallel implementation of the MHN that allows for loading key/value pairs into memory and later performing softmax-based retrieval using a query:
- `forward(x_c, x_q, context_input)`  
  Runs the contextual key/value storage and query-based retrieval phases.
- `__context_forward_step` / `__query_forward_step`  
  Internal routines implementing storage and recall, respectively.
- `reinst_context_forward(x_K_reinst)`  
  Does a forward pass through the MHN using a reinstated key representation.

#### `OneWinnerMHNLayer`
Wraps the MHN into a Transformer-like layer that:
- Projects each input token through different layers of weights to obtain embedded key, value, and/or query representations.
- Loads each set of {input token, key, value} into the MHN during the context window, and subsequently retrieves a softmax-weighted value and reinstated input vector for the query item.

#### Training Pipelines
- `train_mhn_tf_model_batchmode`  
  Training routine for any instantiation of the MHN-Transformer, allowing for various approaches to training the weights \( W_Q, W_K, W_V \). Generally, \( W_Q )\ and \( W_V )\ are trained via a simple backpropagation procedure (but without any backpropagation of information through time), and approaches to training \( W_K )\ vary, as discussed in our paper.
  Supports multiple modes:
  - `K_grad_type ∈ {‘version_1’ (training W_K via MHN), ‘supervised’ (training W_K via supervised query-key alignment), ‘Hebbian’ (training W_K via a Hebbian update)}`
  - `WV_train_mode ∈ {‘via_reinstatement’ (training W_V using reinstated context item; default), ‘via_MHN_output’ (using the MHN output value to train W_V via a delta rule calculation)}`

- `train_mhn_tf_model_batchmode_fixedK`  
  A modified training routine to handle the MHN-Transformer variant in which the key weights \( W_K \) are kept frozen (to their values at initialization).

- `run_case_sequence_model_sweep`  
  Performs multi-trial training runs of any given MHN-Transformer variant (`'tf'` (baseline model), `'mhn_tf_fixed_WK'` (Fixed W_K MHN-Transformer), `'mhn_tf_V1'` (MHN-Transformer with learnable W_K)), collecting accuracy, loss, and covariance statistics for the query, key, and value weights.

---

## Dependencies

Ensure the following dependencies are available:

- numpy  
- matplotlib  
- torch 
- pickle  

---

## Reproducing Our Results

Below are two sample scripts that document how to train our MHN-Transformer models (as well as the baseline Transformer). The first documents how to train a sample MHN-Transformer on the case sequence task; this kind of routine may be used to generate SI Appendix Figs. S8, S9, S11, and S12. The second example documents how to conduct a systematic multi-trial training sweep for a given model; this routine can be used to generate Fig. 8 of the main text and SI Appendix Fig. S10.
These scripts can be executed from within a Python environment.

### **Example 1 — Training a Single MHN-Transformer on the Case-Sequence Task**

This example describes how to train a standalone OneWinnerMHNLayer (a MHN-based Transformer block) on the case-sequence prediction task.  
The task requires the model to infer the case (uppercase/lowercase) of a queried letter within a short sequence.

```python
import torch
import matplotlib.pyplot as plt
from utils import *
from mhn_tf import OneWinnerMHNLayer, train_mhn_tf_model_batchmodefrom dataset import generate_case_sequences

# --- Device setup ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define model parameters ---
num_letters = 4
full_seq_len = 5
k_dim = 50    # used 10 for the baseline Transformer

model_type = 'mhn_tf'    # if baseline Transformer, use 'tf'
tf_dim = 10           # number of MHN hidden neurons; irrelevant if using baseline Transformer
debug_mode = True     # 'debug_mode = True' corresponds to having input projections; irrelevant if using baseline Transformer
criterion = mse_loss
num_batches = 5_000
batch_size = 64
lr = 5e-3            # better to use 1e-3 for baseline Transformer
K_lr = 1e-4          # using separate learning rate for W_K; set to None by default, and irrelevant if using baseline Transformer
K_grad_type = 'supervised'             # train model by supervised query-key alignment; irrelevant if using baseline Transformer 
WV_train_mode = 'via_reinstatement'    # use context item reinstatement during the query time step to train W_V; irrelevant if using baseline Transformer
manual_grad_calc = True    # could set this to 'False' especially for the baseline Transformer, which fully uses automatic differentiation
input_dim = 3 * num_letters
output_dim = 2
dataset_params = ['case_sequence', num_letters]

# --- Initialize a simple MHN-Transformer model architecture ---
model = OneWinnerMHNLayer(batch_size, input_dim, k_dim, output_dim, tf_dim,
                              debug_mode=debug_mode, device=device).to(device)

# if using baseline "minimal" Transformer, use the following line of code instead:
# model = SimplifiedTransformerLayer(input_dim, k_dim, 1, output_dim).to(device)


# --- Train the model on a small version of the case sequence task, using manually computed gradients (instead of automatic gradients) ---
Q_losses, K_losses, V_losses, batch_accs, wv, ul_cov, qk_submat = train_mhn_tf_model_batchmode(model, full_seq_len, dataset_params, criterion,
                             num_batches=num_batches, batch_size=batch_size, lr=lr, K_lr=1e-4,
                             freeze_K=False, freeze_Q=False, freeze_V=False,
                             manual_grad_calc=manual_grad_calc, plot_mode=True, full_key_covar=True,
                             device=device, K_grad_type=K_grad_type, WV_train_mode=WV_train_mode)
batch_losses = Q_losses

# if using baseline "minimal" Transformer, use the following code instead:
# model = SimplifiedTransformerLayer(input_dim, k_dim, n_heads=1, output_dim=output_dim)

# batch_losses, batch_accs, wv, ul_cov, qk_submat = train_tf_batchmode(model, full_seq_len, dataset_params, criterion,
#                    num_batches=num_batches, batch_size=batch_size, lr=lr,
#                    freeze_K=False, freeze_Q=False, freeze_V=False, manual_grad_calc=manual_grad_calc,
#                    device=device, plot_mode=False)

# --- Visualize accuracy and loss curves from training ---
plt.figure()
plt.plot(accs)
plt.xlabel('Batch')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy across Batches')
plt.show()

plt.figure()
plt.plot(batch_losses)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss across Iterations of Training')
plt.show()

# Visualize the learned query, key, value weights and associated covariance matrices
_, _ = visualize_QKV_matrices(model, model_type, label='Final Learned Weights', plot_mode=True, W_V_lims=[-0.2, 1.2, 0.2], QK_lims=[-2, 5, 1])
```

### Example 2 - Performing a Multi-trial Model Sweep

This example describes how to perform multiple training runs of a MHN-Transformer model; here, we specifically focus on the case of training an MHN-Transformer by supervised query-key alignment, though the specific training method can easily be modified. The task is the case sequence task, as before.

```python
import torch
import matplotlib.pyplot as plt
from utils import *
from mhn_tf import run_case_sequence_model_sweep
from dataset import generate_case_sequences

# --- Device setup ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define sweep parameters ---
ntrials = 10
model_type = 'mhn_tf_V1'    # 'tf' if baseline Transformer; 'mhn_tf_fixed_WK' if MHN-Transformer with frozen W_K
num_letters = 4
full_seq_len = 5
k_dim = 50
tf_dim = 16           # number of MHN hidden neurons; irrelevant if using baseline Transformer
debug_mode = True     # 'debug_mode = True' corresponds to having input projections; irrelevant if using baseline Transformer
criterion = mse_loss
num_batches = 5_000
batch_size = 64
lr = 5e-3            # better to use 1e-3 for baseline Transformer
K_lr = 1e-4          # using separate learning rate for W_K; set to None by default, and irrelevant if using baseline Transformer
K_grad_type = 'supervised'            # train model by supervised query-key alignment; 'version_1' if training W_K via MHN, 'none' if not training W_K or if using baseline Transformer model
WV_train_mode = 'via_reinstatement'   # irrelevant if using baseline Transformer model
final_window = 500        # calculate final accuracy statistics by averaging over the final 'final_window' number of iterations of training
save_dir = '[INSERT PATH NAME]/'

# --- Run sweep ---
results_dict, all_accs, all_losses, covar_stats_dict, mean_covar_stats_dict = run_case_sequence_model_sweep(ntrials, model_type, num_letters, full_seq_len, k_dim, tf_dim,
                                                                                                            debug_mode, criterion, num_batches, batch_size, lr, manual_grad_calc=True,
                                                                                                            K_lr=K_lr, K_grad_type=K_grad_type, final_window=final_window, device=device, 
                                                                                                            save_dir=save_dir, WV_train_mode=WV_train_mode)

```


## Citation

If you use or adapt this code, please cite the associated manuscript:

[TO-DO: ADD CITATION UPON PUTTING UP PREPRINT ON ARXIV / SUBMITTING TO A JOURNAL]


## Contact

For questions about code usage or reproducing results, please contact:

Shaunak Bhandarkar -- shaunak@princeton.edu

James McClelland -- jlmcc@stanford.edu

