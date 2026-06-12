import numpy as np
import sys
import random
import argparse
import matplotlib.pyplot as plt
from utils import shuffleData
from data import generate_data, NestedTreeNode, generate_specific_dataset
from kwinnernet import KWinnerNet
from globals import *

def track_weight_asymmetry(kwinner_net, num_samples=1000):
    kwinner_asymmetries = np.zeros(num_samples+1)

    kwinner_asymmetries[0] = kwinner_net.get_weight_asymmetry()

    length = kwinner_net.input_size
    num_active = int(length * kwinner_net.input_sparsity)
    print(length, num_active)
    data = generate_data(num_samples, length, num_active)

    for i in range(num_samples):
        # generate a random pattern and update the networks with it
        x = data[i].reshape(-1, 1)
        _ = kwinner_net.forward(x, phase="learning")

        # track the weight asymmetry of each network after this update
        kwinner_asymmetries[i] = kwinner_net.get_weight_asymmetry()

        if (i+1) % 100 == 0:
            print("Iter " + str(i+1) + ' of ' + str(num_samples) + ' done', flush=True)

    return kwinner_asymmetries

def track_multi_trial_weight_asymmetry(runset, num_mems=1000, num_runs=10):
    all_asymmetries = np.zeros((num_runs, num_mems+1))

    # unpack runset
    n_i, n_h, num_active, f, k, epsilon = runset

    for t in range(num_runs):
        print(f'Trial {t+1} of {num_runs}...', flush=True)
        kwinner_net = KWinnerNet(input_size=n_i, hidden_size=n_h, input_sparsity=1.*num_active/n_i, fan_in_ratio=f, k=k, eta=epsilon)
        asymmetries = track_weight_asymmetry(kwinner_net, num_mems)
        all_asymmetries[t] = asymmetries

    return all_asymmetries


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_samples', type=int, default=1000)
    args = ap.parse_args()

    # # initialize a K-Winner Net with the same parameters as in the main comparison
    # kwinner_net = KWinnerNet(input_size=1000, hidden_size=2000, input_sparsity=0.1, fan_in_ratio=0.05, k=50, eta=0.3)

    # # track the weight asymmetry of the K-Winner Net over the course of learning random patterns
    # kwinner_asymmetries = track_weight_asymmetry(kwinner_net, num_samples=args.num_samples)
    all_asymmetries = track_multi_trial_weight_asymmetry(runset=(1000, 2000, 100, 0.05, 50, 0.3), num_mems=args.num_samples, num_runs=10)

    # plot the resulting weight asymmetry trajectories
    plt.plot(all_asymmetries.T, color='blue', alpha=0.3)
    plt.plot(np.mean(all_asymmetries, axis=0), color='red', label='K-Winner Net Mean')
    plt.xlabel('Iteration')
    plt.ylabel('Weight Asymmetry')
    plt.title('K-Winner Net Weight Asymmetry Trajectory')
    plt.savefig('kwinner_weight_asymmetry.png')
    plt.show()