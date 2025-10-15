# slot_free_neural_computation
Code used to run all simulations and generate all figures for the paper "Neural Computation Without Slots: Steps Towards Biologically Plausible Memory and Attention in Natural and Artificial Intelligence."


# K-winner Modern Hopfield Network (K-winner MHN)
### Supplementary Code for “Slot-Free Neural Computation”

This repository contains code for the *K-winner Modern Hopfield Network* (K-winner MHN), a distributed associative memory model developed as part of the broader *Slot-Free Neural Computation* framework.  

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

