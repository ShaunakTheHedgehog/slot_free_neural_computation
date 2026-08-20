'''
Analysis stage of the K-winner MHN pipeline.

The expensive simulations live in kwinner_mhn_comparison.py, which saves the raw retrieval results
(integer overlaps between each pattern and its retrieved output) via 'run_analysis --analysis raw'.
This file loads one of those raw files and derives the summary curves across memory age -- retrieval
accuracy, d', raw differences and AUC -- then optionally saves and plots them. Since none of this is
compute intensive, curves can be recomputed and replotted freely without re-running any simulation.

Every curve is returned with shape (num_samples, num_mems): one row per independent sample, so that
the mean and SEM across samples are taken the same way for all four metrics.

Example:
    python retrieval_analyses.py --raw_file retrieval_data/raw_correlated_....pkl \
        --metrics dprime,auc --plot
'''

import os
import argparse
import pickle as pkl
from scipy.stats import norm

import numpy as np

from utils import auc_trapezoid
from kwinner_mhn_comparison import plot_data, plot_acc_curves

METRICS = ('accuracy', 'dprime', 'rawdiff', 'auc', 'dprime_auc')


def load_raw(path):
    '''Load a raw retrieval file saved by run_raw_collection.'''
    full_path = path if path.endswith('.pkl') else f'{path}.pkl'
    with open(full_path, 'rb') as fh:
        return pkl.load(fh)


'''
Compute one AUC per memory age, comparing real against pseudo retrieval performance.

At a given age the two inputs each supply 'num_runs' integer overlaps, one per realization. Those are
turned into histograms over the possible overlap values (0...num_active) and handed to auc_trapezoid,
which integrates the resulting ROC. An AUC of 1 means the trained patterns were perfectly separable
from the untrained ones at that age; 0.5 means indistinguishable.

Arguments:
real_overlaps, pseudo_overlaps  :   integer overlaps, each of shape (num_runs, num_mems)
num_active                      :   number of 1-bits per pattern (sets the histogram width)

Returns: a 1D array of AUCs of length num_mems
'''
def aucs_across_ages(real_overlaps, pseudo_overlaps, num_active):
    num_mems = real_overlaps.shape[1]
    aucs = np.zeros(num_mems)
    for age in range(num_mems):
        real_hist = np.bincount(real_overlaps[:, age].astype(int), minlength=num_active + 1)
        pseudo_hist = np.bincount(pseudo_overlaps[:, age].astype(int), minlength=num_active + 1)
        aucs[age] = auc_trapezoid(real_hist, pseudo_hist)
    return aucs


def auc_based_dprimes_across_ages(aucs, max_val=0.9999):
    # convert AUCs to d' values using the inverse of the normal CDF
    # clip to avoid infinite d' values at AUC=0 or 1
    clipped_aucs = np.clip(aucs, 1 - max_val, max_val)
    return np.sqrt(2) * norm.ppf(clipped_aucs)


# mean retrieval accuracy within each sample (shape: num_samples x num_mems)
def accuracy_curves(overlaps, num_active):
    return np.mean(1. * overlaps / num_active, axis=1)


# raw difference between real and pseudo retrieval accuracy, per sample
def rawdiff_curves(real, pseudo, num_active):
    return np.mean(1. * real / num_active, axis=1) - np.mean(1. * pseudo / num_active, axis=1)


'''
d' per sample, matching the definition used in get_match_probabilities: the difference between the
mean real and mean pseudo retrieval accuracy, divided by the standard deviation of their paired
difference across the realizations within that sample.
'''
def dprime_curves(real, pseudo, num_active):
    real_acc = 1. * real / num_active
    pseudo_acc = 1. * pseudo / num_active
    diff = np.mean(real_acc, axis=1) - np.mean(pseudo_acc, axis=1)
    std = np.std(real_acc - pseudo_acc, axis=1)
    return diff / std


# AUC per sample (shape: num_samples x num_mems)
def auc_curves(real, pseudo, num_active):
    return np.stack([aucs_across_ages(real[i], pseudo[i], num_active) for i in range(real.shape[0])])


# d' derived from AUC per sample (shape: num_samples x num_mems)
def auc_based_dprime_curves(real, pseudo, num_active):
    aucs = auc_curves(real, pseudo, num_active)
    return np.stack([auc_based_dprimes_across_ages(aucs[i]) for i in range(aucs.shape[0])])


'''
Derive all requested curves from a raw retrieval file.

Arguments:
raw         :   the dictionary returned by run_raw_collection (or load_raw)
metrics     :   which metrics to compute, any subset of METRICS

Returns a dictionary of (num_samples x num_mems) arrays. Keys follow the naming already used by
run_comparison_test, with '_unif_' variants added when the run included a uniform baseline.
'''
def compute_curves(raw, metrics=METRICS):
    for m in metrics:
        assert m in METRICS, f'unknown metric: {m}'
    num_active = raw['meta']['num_active']
    curves = {}

    models = [('kwinner', 'kwin')]
    if raw['mhn_real'] is not None:
        models.append(('mhn', 'mhn'))

    for label, prefix in models:
        real, pseudo, unif = raw[f'{prefix}_real'], raw[f'{prefix}_pseudo'], raw[f'{prefix}_unif']

        if 'accuracy' in metrics:
            curves[f'{prefix}_out_accs'] = accuracy_curves(real, num_active)
            curves[f'{prefix}_pseudo_out_accs'] = accuracy_curves(pseudo, num_active)
        if 'dprime' in metrics:
            curves[f'{label}_dprimes'] = dprime_curves(real, pseudo, num_active)
        if 'rawdiff' in metrics:
            curves[f'{label}_rawdiffs'] = rawdiff_curves(real, pseudo, num_active)
        if 'auc' in metrics:
            curves[f'{label}_aucs'] = auc_curves(real, pseudo, num_active)
        if 'dprime_auc' in metrics:
            curves[f'{label}_dprimes_auc'] = auc_based_dprime_curves(real, pseudo, num_active)

        # the same metrics measured against the uniform, unstructured pseudo-pattern baseline
        if unif is not None:
            if 'accuracy' in metrics:
                curves[f'{prefix}_unif_pseudo_out_accs'] = accuracy_curves(unif, num_active)
            if 'dprime' in metrics:
                curves[f'{label}_unif_dprimes'] = dprime_curves(real, unif, num_active)
            if 'rawdiff' in metrics:
                curves[f'{label}_unif_rawdiffs'] = rawdiff_curves(real, unif, num_active)
            if 'auc' in metrics:
                curves[f'{label}_unif_aucs'] = auc_curves(real, unif, num_active)
            if 'dprime_auc' in metrics:
                curves[f'{label}_unif_dprimes_auc'] = auc_based_dprime_curves(real, unif, num_active)

    curves['meta'] = raw['meta']
    return curves


# mean and standard error across the independent samples, for a (num_samples x num_mems) curve
def mean_and_sem(curve):
    return np.mean(curve, axis=0), np.std(curve, axis=0) / np.sqrt(curve.shape[0])

def median(curve):
    return np.median(curve, axis=0)


'''
Plot the requested curves. d', raw differences and AUC are drawn with plot_data (which flips the age
axis and shades the SEM); retrieval accuracies are drawn with plot_acc_curves.

Arguments:
curves          :   output of compute_curves
metrics         :   which metrics to plot
plot_dir        :   directory for the saved figures (created if absent)
prefix          :   filename prefix for the figures
uniform         :   plot the uniform-baseline variants instead of the standard ones
max_age         :   maximum memory age to show
ylim, ylim_bottom : vertical limits passed through to plot_data
'''
def plot_curves(curves, metrics=METRICS, plot_dir='analysis_plots', prefix='curves',
                uniform=False, max_age=1000, ylim=None, ylim_bottom=None):
    os.makedirs(plot_dir, exist_ok=True)
    meta = curves['meta']
    runsets = (meta['runset1'], meta['runset2'])
    tag = '_unif' if uniform else ''

    has_mhn = any(k.startswith('mhn') for k in curves)
    assert has_mhn, 'plotting compares two models; re-run the raw collection with --runset2'

    specs = {'dprime': (f'kwinner{tag}_dprimes', f'mhn{tag}_dprimes', "d'"),
             'rawdiff': (f'kwinner{tag}_rawdiffs', f'mhn{tag}_rawdiffs', 'Raw Difference'),
             'auc': (f'kwinner{tag}_aucs', f'mhn{tag}_aucs', 'AUC'),
             'dprime_auc': (f'kwinner{tag}_dprimes_auc', f'mhn{tag}_dprimes_auc', "d' (AUC-based)")}

    for metric in metrics:
        if metric == 'accuracy':
            if 'kwin_out_accs' not in curves or f'kwin{tag}_pseudo_out_accs' not in curves:
                continue
            data = []
            for p in ('kwin', 'mhn'):
                accs = np.mean(curves[f'{p}_out_accs'], axis=0)
                pseudo_accs = np.mean(curves[f'{p}{tag}_pseudo_out_accs'], axis=0)
                unif_accs = (np.mean(curves[f'{p}_unif_pseudo_out_accs'], axis=0)
                             if f'{p}_unif_pseudo_out_accs' in curves else None)
                data.append((accs, pseudo_accs, unif_accs))
            plot_acc_curves(data, runsets, os.path.join(plot_dir, f'{prefix}_accuracy'),
                            max_age=max_age, figsize=(11, 8),
                            uniform_baseline=(data[0][2] is not None), legend=True)
            continue

        kwin_key, mhn_key, ylabel = specs[metric]
        if kwin_key not in curves or mhn_key not in curves:
            continue
        if metric == 'dprime_auc':
            show_median = True
        else:
            show_median = False
        if ylim is None and metric == 'dprime_auc':
            ylim = 5.0
        plot_data(curves[kwin_key], curves[mhn_key],
                  os.path.join(plot_dir, f'{prefix}_{metric}{tag}.png'),
                  max_age=max_age, plot_ylabel=ylabel, cue_level=meta['cue_level'],
                  ylim=ylim, ylim_bottom=ylim_bottom, figsize=(11, 8), show_median=show_median)


def main():
    ap = argparse.ArgumentParser(description='Derive retrieval accuracy / d\' / raw difference / AUC curves from saved raw retrieval data.')
    ap.add_argument('--raw_file', type=str, required=True,
                    help='path to a raw retrieval .pkl saved by run_raw_collection')
    ap.add_argument('--metrics', type=str, default='accuracy,dprime,rawdiff,auc,dprime_auc',
                    help=f'comma-separated subset of {",".join(METRICS)}')
    ap.add_argument('--out_dir', type=str, default='analysis_results',
                    help='directory for the saved curve .pkl')
    ap.add_argument('--filename', type=str, default=None,
                    help='name for the saved curve .pkl (defaults to the raw filename with a curves_ prefix)')
    ap.add_argument('--no_save', action='store_true', help='do not write the curve .pkl')
    ap.add_argument('--plot', action='store_true', help='also produce figures')
    ap.add_argument('--plot_dir', type=str, default='analysis_plots')
    ap.add_argument('--uniform', action='store_true',
                    help='plot the uniform-baseline variants of the curves')
    ap.add_argument('--max_age', type=int, default=1000)
    ap.add_argument('--ylim', type=float, default=None, help='upper y limit for plot_data')
    ap.add_argument('--ylim_bottom', type=float, default=None, help='lower y limit for plot_data')
    args = ap.parse_args()

    metrics = tuple(m.strip() for m in args.metrics.split(',') if m.strip())
    raw = load_raw(args.raw_file)
    print(f'loaded {args.raw_file}', flush=True)

    curves = compute_curves(raw, metrics=metrics)
    print(f'computed curves: {sorted(k for k in curves if k != "meta")}', flush=True)

    base = args.filename or 'curves_' + os.path.basename(args.raw_file).replace('.pkl', '')
    if not args.no_save:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, f'{base}.pkl')
        with open(out_path, 'wb') as fh:
            pkl.dump(curves, fh)
        print(f'saved curves to {out_path}', flush=True)

    if args.plot:
        plot_curves(curves, metrics=metrics, plot_dir=args.plot_dir, prefix=base,
                    uniform=args.uniform, max_age=args.max_age,
                    ylim=args.ylim, ylim_bottom=args.ylim_bottom)
        print(f'saved figures to {args.plot_dir}', flush=True)

    return curves


if __name__ == '__main__':
    main()
