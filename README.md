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



