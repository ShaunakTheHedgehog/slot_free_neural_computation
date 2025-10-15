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


Retrieval Accuracy Across Memory Age

The template code below shows how to generate a plot comparing the retrieval accuracy (across memory ages) for a K-winner MHN and a parameter-matched Original MHN, using 100% and 50% cues. This template code can be used to generate Fig. 4A and SI Appendix Fig. S2.

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
    filename='fig4A_retrieval',
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
    plot_title='Figure 4A',
    plot_xlabel='Memory Age',
    plot_ylabel='Retrieval Accuracy',
    figsize=(8, 5),
    uniform_baseline=False,    # show uniform, unstructured pseudo-pattern baseline in case training on structured, hierarchical data
    legend=False
)

```
