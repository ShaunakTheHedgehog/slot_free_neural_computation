'''
Run the baseline Transformer or an MHN-based Transformer on the case sequence task.

Each call runs `ntrials` independent trials of one model configuration via
run_case_sequence_model_sweep and saves everything to

    <results_root>/<experiment>/<model>/<run>/
        config.json            all arguments + provenance (git commit, host, runtime)
        full_results_*.pkl     the dictionary saved by run_case_sequence_model_sweep

Example:
    python main.py --model QK_proj --L 4 --k_dim 32 --tf_dim 8 --lr 5e-3 --K_lr 1e-4 \
                   --num_batches 5000 --ntrials 10 --experiment paper_L4

Use --dry_run to check arguments and print the output directory without training,
and --demo to run the single-model interactive demo (with plots) instead.
'''
import os
# headless plotting on cluster nodes (must be set before matplotlib is imported)
if 'SLURM_JOB_ID' in os.environ:
    os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import glob
import json
import platform
import random
import subprocess
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from utils import *
from mhn_tf import run_case_sequence_model_sweep, OneWinnerMHNLayer, train_mhn_tf_model_batchmode, train_mhn_tf_model_batchmode_fixedK
from baseline_tf import SimplifiedTransformerLayer, train_tf_batchmode


# model name -> the settings of run_case_sequence_model_sweep that define it
# ('noproj' = no input projections, 'proj' = input projections, 'item' = item_in_mhn)
MODELS = {
    'tf':              dict(model_type='tf',              K_grad_type='none',        debug_mode=False, item_in_mhn=False),
    'fixed_WK_noproj': dict(model_type='mhn_tf_fixed_WK', K_grad_type='none',        debug_mode=False, item_in_mhn=False),
    'fixed_WK_proj':   dict(model_type='mhn_tf_fixed_WK', K_grad_type='none',        debug_mode=True,  item_in_mhn=False),
    'fixed_WK_item':   dict(model_type='mhn_tf_fixed_WK', K_grad_type='none',        debug_mode=True,  item_in_mhn=True),
    'MHN_WK_noproj':   dict(model_type='mhn_tf',       K_grad_type='through_MHN', debug_mode=False, item_in_mhn=False),
    'MHN_WK_proj':     dict(model_type='mhn_tf',       K_grad_type='through_MHN', debug_mode=True,  item_in_mhn=False),
    'MHN_WK_item':     dict(model_type='mhn_tf',       K_grad_type='through_MHN', debug_mode=True,  item_in_mhn=True),
    'QK_noproj':       dict(model_type='mhn_tf',       K_grad_type='supervised',  debug_mode=False, item_in_mhn=False),
    'QK_proj':         dict(model_type='mhn_tf',       K_grad_type='supervised',  debug_mode=True,  item_in_mhn=False),
    'QK_item':         dict(model_type='mhn_tf',       K_grad_type='supervised',  debug_mode=True,  item_in_mhn=True),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # model and task
    p.add_argument('--model', choices=list(MODELS), help='which model to run')
    p.add_argument('--L', type=int, help='number of letters')
    p.add_argument('--C', type=int, default=None, help='context window length (default: L)')
    p.add_argument('--k_dim', type=int, help='key/query dimension N')
    p.add_argument('--tf_dim', type=int, default=None,
                   help='MHN hidden size (default: C for *_item, 2L for *_proj; required for *_noproj; unused for tf)')
    p.add_argument('--input_proj_strength', type=float, default=1.0,
                   help='strength of the identity input projection (*_proj models only)')
    p.add_argument('--beta', type=float, default=1.0,
                   help='inverse temperature multiplying the attention logits (all models)')

    # training
    p.add_argument('--lr', type=float, help='learning rate for W_Q and W_V (and W_K unless --K_lr is given)')
    p.add_argument('--K_lr', type=float, default=None, help='learning rate for W_K (default: --lr)')
    p.add_argument('--num_batches', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--WV_train_mode', choices=['via_reinstatement', 'via_MHN_output'], default='via_reinstatement')
    p.add_argument('--ntrials', type=int, default=10)
    p.add_argument('--final_window', type=int, default=1000,
                   help='number of final batches averaged for the summary accuracy/loss')
    p.add_argument('--seed', type=int, default=0)

    # bookkeeping
    p.add_argument('--results_root', default='results')
    p.add_argument('--experiment', default='default', help='name of the experiment folder')
    p.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)),
                   help='torch CPU threads (default: SLURM_CPUS_PER_TASK, else 1)')
    p.add_argument('--overwrite', action='store_true', help='rerun even if results already exist')
    p.add_argument('--dry_run', action='store_true', help='validate arguments and print the run directory only')
    p.add_argument('--demo', action='store_true', help='run the interactive single-model demo instead')

    args = p.parse_args()
    if args.demo:
        return args

    for name in ['model', 'L', 'k_dim', 'lr']:
        if getattr(args, name) is None:
            p.error(f'--{name} is required')

    spec = MODELS[args.model]
    args.C = args.L if args.C is None else args.C
    if args.C > args.L:
        p.error('C must be <= L (each letter appears at most once in a context window)')

    # hidden size
    if spec['model_type'] == 'tf':
        args.tf_dim = None
    elif args.tf_dim is None:
        if args.model.endswith('_item'):
            args.tf_dim = args.C
        elif args.model.endswith('_proj'):
            args.tf_dim = 2 * args.L
        else:
            p.error(f'--tf_dim is required for {args.model}')

    # only models whose W_K is trained have a K learning rate
    if spec['K_grad_type'] in ['through_MHN', 'supervised']:
        args.K_lr = args.lr if args.K_lr is None else args.K_lr
    else:
        args.K_lr = None

    return args


def run_name(args):
    '''Folder name for one run: every setting that changes the results, so runs never overwrite each other.'''
    parts = [f'L{args.L}', f'C{args.C}', f'k{args.k_dim}']
    if args.tf_dim is not None:
        parts.append(f'h{args.tf_dim}')
    parts.append(f'lr{args.lr:g}')
    if args.K_lr is not None:
        parts.append(f'Klr{args.K_lr:g}')
    parts += [f'nb{args.num_batches}', f'bs{args.batch_size}', f'nt{args.ntrials}']
    if args.model.endswith('_proj') and args.input_proj_strength != 1.0:
        parts.append(f'ips{args.input_proj_strength:g}')
    if args.beta != 1.0:
        parts.append(f'beta{args.beta:g}')
    if args.WV_train_mode != 'via_reinstatement':
        parts.append('WV-MHNout')
    parts.append(f's{args.seed}')
    return '_'.join(parts)


def git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=here,
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', '*.py'], cwd=here,
                                        stderr=subprocess.DEVNULL).decode().strip()
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def run_sweep(args):
    spec = MODELS[args.model]
    run_dir = os.path.join(args.results_root, args.experiment, args.model, run_name(args))

    print(f'run directory: {run_dir}')
    if args.dry_run:
        return

    if glob.glob(os.path.join(run_dir, 'full_results_*.pkl')) and not args.overwrite:
        print('results already exist; skipping (use --overwrite to rerun)')
        return
    os.makedirs(run_dir, exist_ok=True)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = {k: v for k, v in vars(args).items() if k not in ['dry_run', 'demo', 'overwrite']}
    config.update(spec)
    config.update(git_commit=git_commit(), host=platform.node(), torch_version=torch.__version__,
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'))
    config_path = os.path.join(run_dir, 'config.json')

    def write_config(**extra):
        config.update(extra)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    write_config(status='running')
    t0 = time.time()

    sweep_kwargs = dict(K_lr=args.K_lr, K_grad_type=spec['K_grad_type'], final_window=args.final_window,
                        device=torch.device('cpu'), manual_grad_calc=True, save_dir=run_dir,
                        WV_train_mode=args.WV_train_mode, item_in_mhn=spec['item_in_mhn'],
                        input_proj_strength=args.input_proj_strength, beta=args.beta)

    results = run_case_sequence_model_sweep(args.ntrials, spec['model_type'], args.L, args.C + 1,
                                            args.k_dim, args.tf_dim, spec['debug_mode'], mse_loss,
                                            args.num_batches, args.batch_size, args.lr, **sweep_kwargs)

    write_config(status='done', runtime_min=round((time.time() - t0) / 60, 2),
                 mean_final_acc=float(np.mean(results['mean_final_accs'])),
                 mean_final_loss=float(np.mean(results['mean_final_losses'])))
    print(f'done in {config["runtime_min"]} min; mean final accuracy {config["mean_final_acc"]:.4f}')


# interactive single-model demo (with plots); run with `python main.py --demo`
def demo():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_letters = 4
    full_seq_len = 5
    k_dim = 32
    tf_dim = 8
    input_proj_strength = 1.0
    debug_mode = True
    item_in_mhn = False
    criterion = mse_loss
    num_batches = 2000
    batch_size = 64
    lr = 1e-3
    K_lr = 1e-4
    K_grad_type = 'supervised'
    WV_train_mode = 'via_reinstatement'
    input_dim = 3 * num_letters
    output_dim = 2
    dataset_params = ['case_sequence', num_letters]

    # model = SimplifiedTransformerLayer(input_dim, k_dim, 1, output_dim).to(device)
    # batch_losses, batch_accs, wv, ul_cov, qk_submat = train_tf_batchmode(model, full_seq_len, dataset_params, criterion,
    #                    regularizer=None, num_batches=num_batches, batch_size=batch_size, lr=lr,
    #                    toy_task_mode=False, reduced=False, freeze_K=False, freeze_Q=False, freeze_V=False, manual_grad_calc=False,
    #                    visualize_QKV_during=False, plot_mode=True, permutation_reduced=False, W_V_fixed=False,
    #                    full_key_covar=True, plot_freq=300, device=torch.device('cpu'))

    # code for training and evaluating a single MHN-transformer model (in batch mode)
    model = OneWinnerMHNLayer(batch_size, input_dim, k_dim, output_dim, tf_dim, input_proj_strength=input_proj_strength,
                              debug_mode=debug_mode, item_in_mhn=item_in_mhn, device=device).to(device)

    batch_losses, batch_accs, wv, ul_cov, qk_submat, _, _, _ = train_mhn_tf_model_batchmode(model, full_seq_len, dataset_params, criterion,
                                 num_batches=num_batches, batch_size=batch_size, lr=lr,
                                 freeze_K=False, freeze_Q=False, freeze_V=False,
                                 manual_grad_calc=True, plot_mode=True, full_key_covar=True,
                                 device=device, K_grad_type=K_grad_type, WV_train_mode=WV_train_mode, K_lr=K_lr)

    # batch_losses, batch_accs, wv, ul_cov, qk_submat = train_mhn_tf_model_batchmode_fixedK(model, full_seq_len, dataset_params, criterion,
    #                                     num_batches=num_batches, batch_size=batch_size, lr=lr, toy_task_mode=False,
    #                                     reduced=False, manual_grad_calc=True, visualize_QKV_during=False,
    #                                     plot_mode=True, permutation_reduced=False, full_key_covar=True,
    #                                     device=torch.device('cpu'), WV_train_mode=WV_train_mode)

    plt.figure()
    plt.plot(batch_accs)
    plt.xlabel('Batch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy across Batches')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(batch_losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss across Batches')
    plt.legend()
    plt.show()

    # visualize learned Q, K, V weights and covariance matrices
    _, _ = visualize_QKV_matrices(model, 'mhn_tf', label='Final Learned Weights', plot_mode=True) #, W_V_lims=[-0.2, 1.2, 0.2], QK_lims=[-2, 5, 1])


def main():
    args = parse_args()
    if args.demo:
        demo()
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
