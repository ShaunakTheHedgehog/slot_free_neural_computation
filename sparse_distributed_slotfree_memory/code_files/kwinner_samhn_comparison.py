import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from utils import generateData, shuffleData, NestedTreeNode
from kwinnernet import KWinnerNet
from sklearn.linear_model import LinearRegression
import pickle as pkl
from scipy.stats import t


'''
Generate trial-averaged retrieval accuracy curves (across memory age) for two different K-winner MHNs.
Main use case: comparing retrieval accuracy curves between a sample K-winner MHN and a parameter-matched 1-winner MHN.

Arguments:
runset1, runset2        :       the two runsets (tuples of initialization params) used for either K-winner MHN
num_trials              :       the number of trials used to generate average retrieval accuracy curves
num_mems                :       the total number of patterns learned and retrieved within any trial
cue_level               :       the cue level for the partial patterns used during retrieval (between 0 and 1, inclusive)
save_data               :       whether to save the retrieval accuracy data
filename                :       name of file to save
two_runsets             :       whether two runsets are being compared, or if just one runset is being used (i.e. runset2 = None)
data_clustered          :       whether the data is structured (or random/unstructured)
num_flips               :       the number of bit flips, if the data is structured
uniform_baseline        :       whether to generate a baseline of unstructured pseudo-patterns, if the data is structured
nonlinearity_type       :       the type of nonlinearity used at the hidden state -- top-k or random-k 

Returns a dictionary with all of the retrieval accuracy data for both runsets, on both the real patterns and untrained pseudo-patterns.
'''
def get_retrieval_probability_comparison(runset1, runset2, num_trials, num_mems, cue_level, save_data=True, filename='retrieval_acc_comparison', two_runsets=True,
                                         data_clustered=False, num_flips=None, uniform_baseline=False, nonlinearity_type='hard_k'):
    assert nonlinearity_type in ['hard_k', 'random']
    out_matches_mean_1, pseudo_out_matches_mean_1, _, _, unif_results_1 = get_match_probabilities(runset1, num_trials, num_mems, cue_level, in_steady_state=True, data_clustered=data_clustered, num_flips=num_flips, uniform_baseline=uniform_baseline, nonlinearity_type=nonlinearity_type)
    
    out_matches_mean_2, pseudo_out_matches_mean_2, unif_results_2 = None, None, None 
    if two_runsets:
        out_matches_mean_2, pseudo_out_matches_mean_2, _, _, unif_results_2 = get_match_probabilities(runset2, num_trials, num_mems, cue_level, in_steady_state=True, data_clustered=data_clustered, num_flips=num_flips, uniform_baseline=uniform_baseline, nonlinearity_type=nonlinearity_type)

    output_dict = {'model1_out' : out_matches_mean_1, 'model1_pseudo_out' : pseudo_out_matches_mean_1,
                   'model2_out' : out_matches_mean_2, 'model2_pseudo_out' : pseudo_out_matches_mean_2,
                   'model1_unif_pseudo_out' : None, 'model2_unif_pseudo_out' : None}

    if uniform_baseline:
        output_dict['model1_unif_pseudo_out'] = unif_results_1['pseudo_out_mean']
        output_dict['model2_unif_pseudo_out'] = unif_results_2['pseudo_out_mean']
    
    if save_data:
        full_filename = f'{filename}.pkl'
        with open(full_filename, 'wb') as file:
            pkl.dump(output_dict, file)

    # # Can plot the output reconstruction accuracy curves:
    # data = [(out_matches_mean_1, pseudo_out_matches_mean_1), (out_matches_mean_2, pseudo_out_matches_mean_2)]
    # runsets = (runset1, runset2)
    
    # plot_title = f'{filename}.png'
    # plot_acc_curves(data, runsets, plot_title, max_age=1000,
    #                 plot_xlabel='Age of Memory', plot_ylabel='Retrieval Accuracy')

    return output_dict

'''
The main internal helper function used to test learning and retrieval of patterns, that can subsequently be used to generate raw difference and d' curves

Arguments:
runset              :       the specific K-winner MHN to use
num_trials          :       the number of trials over which to calculate retrieval accuracy statistics
num_mems            :       the number of memories learned in each trial
cue_level           :       cue level for partial patterns during retrieval (between 0 and 1, inclusive)
in_steady_state     :       if the K-winner MHN should have a "burn-in" phase for learning pre-existing patterns (not tested on)
data_clustered      :       whether the data is structured
num_flips           :       the number of bit flips, if the data is structured
uniform_baseline    :       whether to generate a baseline of unstructured pseudo-patterns, if the data is structured
nonlinearity_type   :       the type of nonlinearity used at the hidden state -- top-k or random-k 

Returns: (multiple 1d arrays of size 'num_mems')
out_matches_mean            :       the trial-averaged retrieval accuracies across memory age for real trained patterns
pseudo_out_matches_mean     :       the trial-averaged retrieval accuracies across memory age for untrained, pseudo patterns
d_prime_out                 :       the trial-averaged d' across memory age
out_diff                    :       the trial-averaged raw differences across memory age
unif_results                :       a dictionary of similarly trial-averaged measurements for uniform random pseudo-patterns, if needed
'''
def get_match_probabilities(runset, num_trials, num_mems, cue_level, in_steady_state=True, data_clustered=False, num_flips=None, uniform_baseline=False, nonlinearity_type='hard_k'):
    assert nonlinearity_type in ['hard_k', 'random']
    (n_i, n_h, s, f, k, epsilon) = runset
    out_matches = np.zeros((num_trials, num_mems))
    pseudo_out_matches = np.zeros((num_trials, num_mems +1))
    unif_pseudo_out_matches = np.zeros((num_trials, num_mems))


    for i in range(num_trials):

        # generate a K-winner MHN with the given runset
        net = KWinnerNet(n_i, n_h, 1. * s /n_i, f, k, epsilon, nonlinearity_type=nonlinearity_type)

        # generate dataset of patterns, whether structured or random
        full_data = None
        if data_clustered:
            num_data = 7 * num_mems
            tree = NestedTreeNode(pattern_input_size=n_i, pattern_sparsity=1.*s/n_i, num_flips=num_flips)
            full_data = tree.get_clustered_data(num_data=num_data)
        else:
            full_data = generateData(3 * num_mems + 1, n_i, s)

        full_data = shuffleData(full_data)
        steady_state_data = full_data[:num_mems]
        data = full_data[num_mems:2 * num_mems]
        pseudodata = full_data[2 * num_mems:3 * num_mems + 1]

        if data_clustered and uniform_baseline is True:
            unif_pseudodata = generateData(num_mems, n_i, s)
            
        # get K-winner MHN into "steady state" by learning pre-existing patterns if needed
        if in_steady_state:
            _, _ = net.learn_patterns(steady_state_data)

        out_learned = np.zeros((num_mems, n_i))
        # pseudo_out_learned = np.zeros((num_mems + 1, n_i))
        # unif_pseudo_out_learned = np.zeros((num_mems, n_i))

        out = net.retrieve(pseudodata[0].reshape((-1, 1)))

        # perform learning step for all memories, one-by-one
        for j in range(num_mems):
            out = net.forward(data[j].reshape((-1, 1)), phase="learning")
            out = net.retrieve(data[j].reshape((-1, 1)))
            out_learned[j] = out.reshape(-1)

            # out = net.retrieve(pseudodata[j + 1].reshape((-1, 1)))
            # pseudo_out_learned[j + 1] = out.reshape(-1)

            # if data_clustered and uniform_baseline is True:
            #     out = net.retrieve(unif_pseudodata[j].reshape((-1, 1)))
            #     unif_pseudo_out_learned[j] = out.reshape(-1)

        # evaluate output pattern completion abilities for both real and pseudo data 
        _, out_retrieve = net.retrieve_from_partial_cues(data, cue_level)  
        _, pseudo_out_retrieve = net.retrieve_from_partial_cues(pseudodata, cue_level)  

        # if the data is structured and the uniform, random pattern is needed, also evaluate pattern completion abilities in this case
        if data_clustered and uniform_baseline is True:
            _, unif_pseudo_out_retrieve = net.retrieve_from_partial_cues(unif_pseudodata, cue_level)
            unif_pseudo_out_matches[i] = 1. * np.sum(unif_pseudodata * unif_pseudo_out_retrieve, axis=1) / s

        out_matches[i] = 1. * np.sum(data * out_retrieve, axis=1) / s
        pseudo_out_matches[i] = 1. * np.sum(pseudodata * pseudo_out_retrieve, axis=1) / s


    out_matches_mean = np.mean(out_matches, axis=0)

    # calculate pseudo-data baseline -- these should all roughly be the same value, since the model is in steady state (i.e. burn-in phase is completed)
    pseudo_out_matches_mean = np.mean(pseudo_out_matches, axis=0)

    # if needed, calculate statistics of the uniform, random, unstructured pseudo-patterns
    unif_pseudo_out_matches_mean = np.mean(unif_pseudo_out_matches, axis=0)
    unif_out_diff = out_matches_mean - unif_pseudo_out_matches_mean
    unif_out_std = np.std(out_matches - unif_pseudo_out_matches, axis=0)
    unif_d_prime_out = unif_out_diff / unif_out_std
    unif_results = {'pseudo_out_mean' : unif_pseudo_out_matches_mean, 
                    'out_diff' : unif_out_diff, 'dprime_out' : unif_d_prime_out}

    # calculate raw differences and d prime sample
    out_diff = out_matches_mean - pseudo_out_matches_mean[:-1]
    out_std = np.std(out_matches - pseudo_out_matches[:, :-1], axis=0)
    d_prime_out = out_diff / out_std

    return out_matches_mean, pseudo_out_matches_mean, d_prime_out, out_diff, unif_results


'''
The main function used to generate d' and raw difference curves, comparing these across two different K-winner MHNs

Arguments:
runset1, runset2        :       the two runsets (tuples of initialization params) used for either K-winner MHN
num_mems                :       the total number of patterns learned and retrieved within any trial
num_samples             :       the total number of d' / raw difference samples to generate
num_runs_per_sample     :       the number of separate K-winner MHN trials used to generate each d' or raw difference sample
cue_level               :       the cue level for the partial patterns used during retrieval (between 0 and 1, inclusive)
data_clustered          :       whether the data is structured
num_flips               :       the number of bit flips, if the data is structured
save_data               :       whether to save the data 
filename                :       name of file to save
two_runsets             :       whether two runsets are being compared, or if just one runset is being used (i.e. runset2 = None)

Returns a dictionary of the relevant d' and raw difference information across both model types.
'''
def run_comparison_test(runset1, runset2, num_mems, num_samples, num_runs_per_sample, cue_level, data_clustered=False, 
                        num_flips=None, save_data=True, filename='kwinner_mhn_comparison_results', two_runsets=True):
    k_winner_dprimes = np.zeros((num_samples, num_mems))
    samhn_dprimes = np.zeros((num_samples, num_mems))
    d_prime_measurements = np.zeros((num_samples, num_mems))

    kwinner_rawdiffs = np.zeros((num_samples, num_mems))
    samhn_rawdiffs = np.zeros((num_samples, num_mems))

    kwin_accs = np.zeros((num_samples, num_mems))
    kwin_pseudo_accs = np.zeros_like(kwin_accs)
    mhn_accs = np.zeros((num_samples, num_mems))
    mhn_pseudo_accs = np.zeros_like(mhn_accs)


    for i in range(num_samples):
        out_matches, pseudo_out_matches, d_out, out_diff = get_match_probabilities(runset=runset1, num_trials=num_runs_per_sample, num_mems=num_mems, cue_level=cue_level, in_steady_state=True, data_clustered=data_clustered, num_flips=num_flips)
        k_winner_dprimes[i] = d_out
        kwinner_rawdiffs[i] = out_diff
        kwin_accs[i] = out_matches
        kwin_pseudo_accs[i] = pseudo_out_matches[:-1]
        
        if two_runsets:
            samhn_out_matches, samhn_pseudo_out_matches, samhn_d_out, samhn_out_diff = get_match_probabilities(runset=runset2, num_trials=num_runs_per_sample, num_mems=num_mems, cue_level=cue_level, in_steady_state=True, data_clustered=data_clustered, num_flips=num_flips)
            samhn_dprimes[i] = samhn_d_out
            d_prime_measurements[i] = d_out - samhn_d_out
            samhn_rawdiffs[i] = samhn_out_diff
            mhn_accs[i] = samhn_out_matches
            mhn_pseudo_accs[i] = samhn_pseudo_out_matches[:-1]        
        

        print("Iter " + str(i+1) + ' of ' + str(num_samples) + ' done')


    results_dict = {'kwinner_dprimes' : k_winner_dprimes, 'mhn_dprimes' : samhn_dprimes,
                    'kwinner_rawdiffs' : kwinner_rawdiffs, 'mhn_rawdiffs' : samhn_rawdiffs,
                    'dprime_difference' : d_prime_measurements,
                    'kwin_out_accs': kwin_accs, 'kwin_pseudo_out_accs': kwin_pseudo_accs, 
                    'mhn_out_accs': mhn_accs, 'mhn_pseudo_out_accs': mhn_pseudo_accs}
    
    if save_data:
        full_filename = f'{filename}.pkl'
        with open(full_filename, 'wb') as file:
            pkl.dump(results_dict, file)

    return results_dict


'''
Run a one-tailed t-test.
Arguments:
t_vals      :       1D array of t-values (i.e. differences) that are being evaluated
df          :       number of degrees of freedom
alpha       :       significance level

Returns:
kwinner_better_inds     :       indices where the K-winner MHN t-values are larger, according to the t-test
samhn_better_inds       :       indices where the 1-winner MHN t-values are larger, according to the t-test
'''
def get_one_tailed_significance_indices(t_vals, df, alpha):
    # One-tailed t-value (right tail)
    t0 = t.ppf(1 - alpha, df)

    kwinner_better_inds = np.argwhere(t_vals > t0).reshape(-1)
    samhn_better_inds = np.argwhere(t_vals < -t0).reshape(-1)

    return kwinner_better_inds, samhn_better_inds


# perform exponential decay regression of a curve stored in a 1D array 'data', 
# using the first 'idxs' number of indices to fit the regression 
def exponential_regress_data(data, idxs, Y_FLOOR=1e-100):
    y = np.maximum(data, Y_FLOOR)
    log_y = np.log(y)
    x = np.arange(1, len(y)+1).reshape((-1, 1))

    reg = LinearRegression().fit(x[:idxs], log_y[:idxs])
    regressed_y = np.exp(reg.coef_ * x + reg.intercept_)
    print(np.exp(reg.intercept_), -reg.coef_[0])
    print("r^2 value up to index " + str(idxs) + ": " + str(reg.score(x[:idxs], log_y[:idxs])))
    return regressed_y


'''
Main function used for plotting d' and raw difference curves, and comparing them between a K-winner MHN and a 1-winner MHN

Arguments:
kwinner_data        :       d' / raw difference data for the K-winner MHN (num_samples x num_mems)
samhn_data          :       d' / raw difference data for the 1-winner MHN (num_samples x num_mems)
plot_title          :       name to give the file storing the figure
max_age             :       maximum memory age to show in plot
plot_xlabel         :       x-axis label
plot_ylabel         :       y-axis label
plot_rel_advantages :       show where K-winner MHN (and 1-winner MHN) values are greater, according to a one-tailed t-test
plot_mode           :       can opt to plot on 'log_y' scale (default: None)
with_regression     :       whether to show regressed exponential decay curves, if plotting raw differences
regression_idxs     :       the maximum memory age used to fit exponential decay regression curves
mhn_runset          :       runset of the 1-winner MHN, if calculating theoretical raw difference curve for MHN
cue_level           :       cue level used during retrieval
plot_main_data      :       whether to plot d' / raw difference curves (default: True)
lw                  :       line width for empirical curves
theory_lw           :       line width for theoretical / regression curves
figsize             :       size of plot
'''
def plot_data(kwinner_data, samhn_data, plot_title, max_age=1000,
              plot_xlabel='100% Cues', plot_ylabel='d\'', plot_rel_advantages=True, plot_mode=None,
              with_regression=False, regression_idxs=200, mhn_runset=None, cue_level=1.0, plot_main_data=True, lw=2.2, theory_lw=2.5, figsize=None):

    kwinner_data = np.flip(kwinner_data, axis=1)

    num_samples = kwinner_data.shape[0]

    samhn_data = np.flip(samhn_data, axis=1)

    mean_kwinner_data = np.mean(kwinner_data, axis=0)
    kwinner_std_error = np.std(kwinner_data, axis=0) / np.sqrt(kwinner_data.shape[0])

    mean_samhn_data = np.mean(samhn_data, axis=0)
    samhn_std_error = np.std(samhn_data, axis=0) / np.sqrt(samhn_data.shape[0])

    if plot_rel_advantages:
        data_diffs = kwinner_data - samhn_data
        t_vals = np.mean(data_diffs, axis=0) / (np.std(data_diffs, axis=0) / np.sqrt(data_diffs.shape[0]))
        kwinner_better_inds, samhn_better_inds = get_one_tailed_significance_indices(t_vals, df=num_samples-1, alpha=0.01)
        kwinner_better_inds = np.asarray(kwinner_better_inds) + 1

        samhn_better_inds = np.asarray(samhn_better_inds) + 1

        np.set_printoptions(threshold=sys.maxsize)

        print(kwinner_better_inds)
        print(samhn_better_inds)

    if with_regression and plot_mode == 'log_y':
        max_age = 350

    ages = np.arange(1, max_age+1)

    if figsize is None:
        plt.figure(figsize=(15, 5))
    else:
        plt.figure(figsize=figsize)


    if plot_main_data:
        plt.plot(ages, mean_kwinner_data[:max_age], label="K-Winner MHN", color='red', lw=lw)
        if not with_regression:
            plt.fill_between(ages, mean_kwinner_data[:max_age] - kwinner_std_error[:max_age], mean_kwinner_data[:max_age] + kwinner_std_error[:max_age], color='cyan')

    if plot_rel_advantages:
        if plot_ylabel == 'Raw Difference':
            plt.scatter(kwinner_better_inds, -0.05*np.ones_like(kwinner_better_inds), s=1, color='red')
        else:
            plt.scatter(kwinner_better_inds, -0.4*np.ones_like(kwinner_better_inds), s=1, color='red')


    if plot_main_data:
        plt.plot(ages, mean_samhn_data[:max_age], label="Original MHN", color='gray', lw=lw)
        if not with_regression:
            plt.fill_between(ages, mean_samhn_data[:max_age] - samhn_std_error[:max_age], mean_samhn_data[:max_age] + samhn_std_error[:max_age], color='cyan')


    if plot_rel_advantages:
        if plot_ylabel == 'Raw Difference':
            plt.scatter(samhn_better_inds, -0.05*np.ones_like(samhn_better_inds), s=1, color='black')
        else:
            plt.scatter(samhn_better_inds, -0.4*np.ones_like(samhn_better_inds), s=1, color='black')


    if with_regression and mhn_runset is not None:
        print("Printing regression constants (C, a) for y = C*exp(-ax)...")
        print("K-Winner regression constants: ")
        regressed_kwinner_data = exponential_regress_data(mean_kwinner_data[:max_age], regression_idxs)
        print("MHN regression constants: ")
        regressed_samhn_data = exponential_regress_data(mean_samhn_data[:max_age], regression_idxs)
        plt.plot(ages, regressed_kwinner_data, color='orange', label="Regressed K-Winner", lw=theory_lw, linestyle='--')
        plt.plot(ages, regressed_samhn_data, color='black', label="Regressed MHN", lw=theory_lw, linestyle='--')


        (n_i, n_h, s, f, k, epsilon) = mhn_runset
        p_baseline = 1.*s/n_i + np.sqrt(2*cue_level/n_i * (1 - s/n_i) * np.log(n_h))
        theoretical_mhn = ((1 - 1/n_h)**(ages - 1)) * (1 - p_baseline)
        print("Theoretical MHN regression constants: " + str(1 - p_baseline) + " " + str(-np.log(1 - 1/n_h)))
        plt.plot(ages, theoretical_mhn, color='green', label="Theoretical MHN", lw=theory_lw, linestyle='dashdot')

    plt.ylabel(plot_ylabel, fontsize=28)
    plt.xlabel(plot_xlabel, fontsize=20)

    if plot_mode == 'log_y':
        plt.yscale('log')

    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)


    if plot_ylabel == 'Raw Difference':
        if plot_mode != 'log_y':
            plt.ylim(bottom=-0.1, top=1.0)

    else:
        plt.ylim(bottom=-0.5, top=5.)
    plt.xlim(0, 1000)

    if not with_regression:
        plt.xticks(np.arange(0, max_age+1, 200))
    else:
        plt.xticks(np.arange(0, 350.1, 50))
        plt.xlim(0, 350)

    if plot_ylabel == 'Raw Difference':
        if plot_mode != 'log_y':
            plt.yticks(np.arange(0, 1.01, 0.2))
    else:
        plt.yticks(np.arange(0, 5.1, 1))


    if not with_regression:
        plt.legend(fontsize=20, loc='upper right')
    else:
        plt.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.7)

    plt.xlabel("Age of Memory", fontsize=20)

    plt.savefig(plot_title)



# def plot_multiple_curves(filepath_list, runsets, plot_title, max_age=1000, colors=None,
#                         plot_xlabel='Age of Memory', plot_ylabel='d\''):
#     ages = np.arange(1, max_age + 1)
#     plt.figure(figsize=(15, 8))

#     for i in range(len(filepath_list)):
#         filepath = filepath_list[i]
#         (n_i, n_h, s, f, k, epsilon) = runsets[i]
#         if colors is not None:
#             color = colors[i]
#         kmhn_data = np.load(filepath)
#         kmhn_data = np.flip(kmhn_data, axis=1)

#         mean_kmhn_data = np.mean(kmhn_data, axis=0)
#         kmhn_std_error = np.std(kmhn_data, axis=0) / np.sqrt(kmhn_data.shape[0])

#         plt.plot(ages, mean_kmhn_data[:max_age], label="epsilon=" + str(epsilon) + ", f=" + str(f) + ", k=" + str(k), color=color)
#         plt.fill_between(ages, mean_kmhn_data[:max_age] - kmhn_std_error[:max_age], mean_kmhn_data[:max_age] + kmhn_std_error[:max_age], color='cyan')

#     plt.ylabel(plot_ylabel, fontsize=32)
#     plt.xlabel(plot_xlabel, fontsize=24)

#     plt.xticks(fontsize=28)
#     plt.yticks(fontsize=28)

#     # plt.ylim(bottom=-0.1, top=0.85)
#     plt.ylim(bottom=-0.5, top=5.)
#     plt.xlim(0, max_age+1)
#     plt.xticks(np.arange(0, max_age+1, 100))

#     plt.yticks(np.arange(0, 5.1, 1))

#     plt.legend(fontsize=20, loc='upper right')

#     plt.savefig(plot_title)


'''
Main function for plotting output retrieval accuracy curves
Main use case: comparing retrieval accuracy curves between a K-winner MHN and a parameter-matched 1-winner MHN.

Arguments:
data                :       a sequence of two tuples of the form (accs, pseudo_accs, uniform pseudo_accs), the first for the K-winner MHN and the second for the 1-winner MHN
runsets             :       the two runsets used (K-winner MHN and 1-winner MHN)
plot_title          :       title of the file used to store the plot
max_age             :       maximum memory age to show in the plot
plot_xlabel         :       x-axis label
plot_ylabel         :       y-axis label
figsize             :       size of plot
uniform_baseline    :       whether to show a uniform, random pseudo-pattern baseline (in the case of training on structured data)
legend              :       whether to show a plot legend 
'''
def plot_acc_curves(data, runsets, plot_title, max_age=1000,
                    plot_xlabel='Age of Memory', plot_ylabel='Retrieval Accuracy', figsize=None,
                    uniform_baseline=False, legend=True):
    assert len(data) == 2
    ages = np.arange(1, max_age + 1)

    if figsize is None:
        plt.figure(figsize=(10, 10))
    else:
        plt.figure(figsize=figsize)

    colors = [('red', 'lightcoral', 'chocolate'), ('gray', 'lightgray', 'lightsteelblue')]
    labels = [('Real Data K-winner MHN', 'Pseudo-data K-winner MHN', 'Uniform Pseudo-data K-winner MHN'), ('Real Data Original MHN', 'Pseudo-data Original MHN', 'Uniform Pseudo-data Original MHN')]

    ndata = len(data)

    for i in range(ndata-1, -1, -1):
        (kmhn_accs, kmhn_pseudo_accs, kmhn_unif_pseudo_accs) = data[i]
        real_color, pseudo_color, unif_pseudo_color = colors[i]
        real_label, pseudo_label, unif_pseudo_label = labels[i]
        

        (n_i, n_h, s, f, k, epsilon) = runsets[i]
        kmhn_accs = np.flip(kmhn_accs)
        kmhn_pseudo_accs = np.flip(kmhn_pseudo_accs)

        if uniform_baseline:
            plt.plot(ages, kmhn_unif_pseudo_accs[:max_age], label=unif_pseudo_label, color=unif_pseudo_color, lw=3)

        plt.plot(ages, kmhn_pseudo_accs[:max_age], label=pseudo_label, color=pseudo_color, lw=3)
        plt.plot(ages, kmhn_accs[:max_age], label=real_label, color=real_color, lw=3)
        # plt.fill_between(ages, mean_kmhn_data[:max_age] - kmhn_std_error[:max_age], mean_kmhn_data[:max_age] + kmhn_std_error[:max_age], color='cyan')

    plt.ylabel(plot_ylabel, fontsize=30)
    plt.xlabel(plot_xlabel, fontsize=28)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    plt.ylim(bottom=-0.05, top=1.05)
    plt.xlim(0, max_age+1)
    
    plt.xticks(np.arange(0, max_age+1, 200))

    plt.yticks(np.arange(0, 1.1, 0.2))

    if legend:
        plt.legend(fontsize=16, loc='upper right')

    plt.savefig(f'{plot_title}.png')


# A template function that one can use to generate retrieval accuracy curves, and compare
# them between a K-winner MHN and 1-winner MHN, in the case of random, unstructured data
def run_random_data_test():
    print('Running retrieval tests on random data... ')

    scaledup_mhn_runset = (1_000, 100, 100, 1., 1, 1.)
    scaledup_kwin_runset = (1_000, 2_000, 100, 0.05, 50, 0.3)
    runsets = (scaledup_kwin_runset, scaledup_mhn_runset)

    num_mems = 2_000

    # num_samples = 10
    # num_runs_per_sample = 20
    num_trials = 100

    cue_level1 = 1.0
    cue_level2 = 0.5
    num_flips = None 
    data_clustered = False
    nonlinearity_type = 'random'

    k = 50
    epsilon = 0.3
    f = 0.05

    filename1 = f'randomdata_cuelevel{cue_level1}_retrieval_accs'
    filename2 = f'randomdata_cuelevel{cue_level2}_retrieval_accs'

    output_dict1 = get_retrieval_probability_comparison(scaledup_kwin_runset, scaledup_mhn_runset, num_trials, num_mems, cue_level1, save_data=True, filename=filename1, two_runsets=True,
                                         data_clustered=data_clustered, num_flips=num_flips, uniform_baseline=False, nonlinearity_type=nonlinearity_type)

    output_dict2 = get_retrieval_probability_comparison(scaledup_kwin_runset, scaledup_mhn_runset, num_trials, num_mems, cue_level2, save_data=True, filename=filename2, two_runsets=True,
                                         data_clustered=data_clustered, num_flips=num_flips, uniform_baseline=False, nonlinearity_type=nonlinearity_type)

    # with open(f'{filename1}.pkl', 'rb') as file1:
    #     output_dict1 = pkl.load(file1)
    
    # with open(f'{filename2}.pkl', 'rb') as file2:
    #     output_dict2 = pkl.load(file2)

    print(output_dict1)
    print(output_dict2)

    data1 = [(output_dict1['model1_out'], output_dict1['model1_pseudo_out'], output_dict1['model1_unif_pseudo_out']), (output_dict1['model2_out'], output_dict1['model2_pseudo_out'], output_dict1['model2_unif_pseudo_out'])]
    plot_acc_curves(data1, runsets, f'{filename1}', figsize=(8, 5), uniform_baseline=False, legend=False)
     

    data2 = [(output_dict2['model1_out'], output_dict2['model1_pseudo_out'], output_dict2['model1_unif_pseudo_out']), (output_dict2['model2_out'], output_dict2['model2_pseudo_out'], output_dict2['model2_unif_pseudo_out'])]
    plot_acc_curves(data2, runsets, f'{filename2}', figsize=(8, 5), uniform_baseline=False, legend=False)


# A template function one can use to generate retrieval accuracy curves in the case of structured data
def run_tree_data_test():
    scaledup_mhn_runset = (1_000, 100, 100, 1., 1, 1.)
    scaledup_kwin_runset = (1_000, 2_000, 100, 0.05, 50, 0.3)
    runsets = (scaledup_kwin_runset, scaledup_mhn_runset)

    num_mems = 2_000

    # num_samples = 10
    # num_runs_per_sample = 20
    num_trials = 100

    cue_level1 = 1.0
    cue_level2 = 0.5
    num_flips = 30 
    data_clustered = True

    k = 50
    epsilon = 0.3
    f = 0.05

    filename1 = f'treedata_cuelevel{cue_level1}_numflips{num_flips}_retrieval_accs'
    filename2 = f'treedata_cuelevel{cue_level2}_numflips{num_flips}_retrieval_accs'
    pathname = 'treedata_acc_curves'

    # output_dict1 = get_retrieval_probability_comparison(scaledup_kwin_runset, scaledup_mhn_runset, num_trials, num_mems, cue_level1, save_data=True, filename=filename1, two_runsets=True,
    #                                      data_clustered=data_clustered, num_flips=num_flips, uniform_baseline=True)

    # output_dict2 = get_retrieval_probability_comparison(scaledup_kwin_runset, scaledup_mhn_runset, num_trials, num_mems, cue_level2, save_data=True, filename=filename2, two_runsets=True,
    #                                      data_clustered=data_clustered, num_flips=num_flips, uniform_baseline=True)

    with open(f'{filename1}.pkl', 'rb') as file1:
        output_dict1 = pkl.load(file1)
    
    with open(f'{filename2}.pkl', 'rb') as file2:
        output_dict2 = pkl.load(file2)

    data1 = [(output_dict1['model1_out'], output_dict1['model1_pseudo_out'], output_dict1['model1_unif_pseudo_out']), (output_dict1['model2_out'], output_dict1['model2_pseudo_out'], output_dict1['model2_unif_pseudo_out'])]
    plot_acc_curves(data1, runsets, f'{pathname}/{filename1}', figsize=(8, 5), uniform_baseline=True, legend=False)
     

    data2 = [(output_dict2['model1_out'], output_dict2['model1_pseudo_out'], output_dict2['model1_unif_pseudo_out']), (output_dict2['model2_out'], output_dict2['model2_pseudo_out'], output_dict2['model2_unif_pseudo_out'])]
    plot_acc_curves(data2, runsets, f'{pathname}/{filename2}', figsize=(8, 5), uniform_baseline=True, legend=True)


# The main function used to generate d' and raw difference curves.
# Can also be used to generate raw retrieval accuracy curves.
def main():

    print("Running d primes / raw differences test on random data...")

    scaledup_mhn_runset = (1_000, 100, 100, 1., 1, 1.)

    scaledup_kwin_runset = (1_000, 2_000, 100, 0.05, 50, 0.3)
    new_scaledup_kwin_runset = (1_000, 1_000, 100, 0.1, 50, 0.3)


    # CODE FOR GENERATING d' / RAW DIFF PLOTS
    num_mems = 2_000

    num_samples = 10
    num_runs_per_sample = 20
    cue_level = 1.0
    # cue_level2 = 0.5
    num_flips = None 
    data_clustered = False

    k = 50
    epsilon = 0.3
    f = 0.05


    filename = f'kwinner_mhn_scaledup_comparison_random_data_eps={epsilon}_f={f}_k={k}_{num_samples}samples_{num_runs_per_sample}runs_cuelevel{cue_level}_{num_flips}flips'
    full_filename = f'{filename}.pkl'

    runsets = (new_scaledup_kwin_runset, scaledup_mhn_runset)


    # Run the test to generate d' / raw difference curve statistics
    res_dict = run_comparison_test(new_scaledup_kwin_runset, scaledup_mhn_runset, num_mems, num_samples=num_samples, num_runs_per_sample=num_runs_per_sample, cue_level=cue_level, data_clustered=data_clustered, num_flips=num_flips, cnn_data=None, save_data=True, filename=filename, two_runsets=False)

    with open(full_filename, 'rb') as file:
        results_dict = pkl.load(file)
    
    kwinner_data = results_dict['kwinner_dprimes']
    samhn_data = results_dict['mhn_dprimes']


    filename = f'basic_dprime_comparison_nummems{num_mems}_numsamples{num_samples}_numrunspersample{num_runs_per_sample}_cuelevel{cue_level}_dataclustered{data_clustered}_numflips{num_flips}'

    plot_title = f'dprimes_{filename}.png'
    plot_data(kwinner_data, samhn_data, plot_title, max_age=1000,
              plot_xlabel='100% Cues', plot_ylabel='d\'', plot_rel_advantages=True, plot_mode=None,
              with_regression=False, regression_idxs=200, mhn_runset=None, cue_level=1.0, plot_main_data=True, lw=2.5, theory_lw=4.0, figsize=(8, 5))
    
    


    
    # CODE FOR GENERATING RETRIEVAL ACCURACY CURVES
    num_trials = 100
    num_mems = 2_000
    cue_level1 = 1.0
    cue_level2 = 0.5
    k = 50
    epsilon = 0.3
    f = 0.05
    filename1 = f'kwin_acc_curve_scaledup_k={k}_eps={epsilon}_f={f}_numtrials{num_trials}_nummems{num_mems}_cuelevel{cue_level1}'
    filename2 = f'kwin_acc_curve_scaledup_k={k}_eps={epsilon}_f={f}_numtrials{num_trials}_nummems{num_mems}_cuelevel{cue_level2}'

    get_retrieval_probability_comparison(new_scaledup_kwin_runset, None, num_trials, num_mems, cue_level=cue_level1, save_data=True, filename=filename1, two_runsets=False)
    get_retrieval_probability_comparison(new_scaledup_kwin_runset, None, num_trials, num_mems, cue_level=cue_level2, save_data=True, filename=filename2, two_runsets=False)


    runsets = (scaledup_kwin_runset, scaledup_mhn_runset)

    # 100% cue level
    with open(f'{filename1}.pkl', 'rb') as file1:
        output_dict1 = pkl.load(file1)
    


    # 50% cue level
    with open(f'{filename2}.pkl', 'rb') as file2:
        output_dict2 = pkl.load(file2)

    
    filename1 = 'acc_curves_cuelevel100'
    data1 = [(output_dict1['model1_out'], output_dict1['model1_pseudo_out']), (output_dict1['model2_out'], output_dict1['model2_pseudo_out'])]
    plot_acc_curves(data1, runsets, f'{filename1}', figsize=(8, 5))
    
    with open(f'{filename2}.pkl', 'rb') as file2:
        output_dict2 = pkl.load(file2) 

    filename2 = 'acc_curves_cuelevel50'
    data2 = [(output_dict2['model1_out'], output_dict2['model1_pseudo_out']), (output_dict2['model2_out'], output_dict2['model2_pseudo_out'])]
    plot_acc_curves(data2, runsets, f'{filename2}', figsize=(8, 5))



if __name__ == "__main__":
    # main()
    # run_random_data_test()
    run_tree_data_test()