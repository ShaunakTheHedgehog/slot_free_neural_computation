import sys
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.optimize as opt
import pandas as pd
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
import copy
import csv
import itertools


'''
Generate a dataset of binary patterns, all with the same sparsity level
Arguments:
num_examples    :     number of patterns to generate  
length          :     total length of each binary pattern
num_active      :     number of 1-bits in each pattern (i.e. effectively the sparsity level)

Returns:
data            :     a 2D array of binary patterns (with shape num_examples x length)
'''
def generateData(num_examples, length, num_active):
  data = np.zeros((num_examples, length))
  for i in range(num_examples):
    data_vec = np.zeros(length)
    rand_indices = random.sample(range(length), num_active)
    data_vec[rand_indices] = 1.
    data[i] = data_vec
  return data


# softmax nonlinearity, with an inverse temperature parameter (i.e. slope_param)
def softmax(x, slope_param):
  denom = np.sum(np.exp((x - np.amax(x)) * slope_param))
  softmax = np.exp((x - np.amax(x)) * slope_param) / denom

  return softmax


# take in any real pattern vector 'x' and sparsify it so that 'sparsity' proportion 
# of its bits are 1 and the rest are 0
def sparsify(x, sparsity): 
  num_active = int(sparsity * len(x))
  x_arr = x.reshape(len(x))
  sorted_indices = np.argsort(x_arr)
  top_indices = sorted_indices[-num_active:]

  sparse_x = np.zeros_like(x)
  sparse_x[top_indices] = 1
  return sparse_x


# apply a hard argmax rule to the real pattern vector x
def one_winner_take_all(x):
  sparse = np.zeros_like(x)
  max_ind = np.argmax(x)
  sparse[max_ind] = 1.
  return sparse


# apply a top-k rule to the real pattern vector 'x', setting the 'k' largest 
# entries to be 1 and the rest to be 0
def hard_k_winner_take_all(x, k):
  sparse = np.zeros_like(x)
  max_inds = np.argsort(x.reshape(-1))
  max_inds = max_inds[-k:]
  sparse[max_inds] = 1
  return sparse


# apply a "random k" rule to the real pattern vector 'x', setting the 'k' largest
# entries to be 1 and the rest to be 0
def random_k_winner_take_all(x, k):
  x_sparse = np.zeros_like(x)
  x_sparse[np.random.choice(len(x), k, replace=False)] = 1.
  return x_sparse

# calculate the mean and std of the diagonal entries in a square matrix
def calculate_diagonal_statistics(square_matrix):
  mean = np.trace(square_matrix) / square_matrix.shape[0]
  diagonal = np.sum(square_matrix * np.eye(square_matrix.shape[0]), axis=0)
  std = np.std(diagonal)
  return mean, std

# calculate the mean and std of the off-diagonal entries of a square matrix
def calculate_offdiagonal_statistics(square_matrix):
  mean = (np.sum(square_matrix) - np.trace(square_matrix)) / (square_matrix.shape[0]**2 - square_matrix.shape[0])
  std = (np.std(square_matrix - np.eye(square_matrix.shape[0]) * (square_matrix - mean))) * np.sqrt( 1.0 * square_matrix.shape[0]**2 / (square_matrix.shape[0]**2 - square_matrix.shape[0]) )
  return mean, std


'''
Perform retrieval from a collection of partial patterns, across a range of partial cue levels
Arguments:
net             :         the specific K-winner MHN model used
col_data        :         sequence of original full patterns, with shape input_size x num_patterns
cue_percentages :         the cue percentages to test out for retrieval

Returns:
dot_prod_accuracies     :   stores average reconstruction accuracies across each cue level (shape is len(cue_percentages))
retrieved_data_matrices :   full collection of retrieved patterns across all patterns in the dataset and all cue_levels (shape is len(cue_percentages) x num_patterns x input_size)
'''
def perform_partial_cue_test(net, col_data, cue_percentages=np.arange(0.05, 1.05, 0.05, dtype=float)):

  dot_prod_accuracies = 1.0 * np.zeros(len(cue_percentages))
  retrieved_data_matrices = np.zeros((len(cue_percentages), col_data.shape[1], col_data.shape[0]))

  for i in range(len(cue_percentages)):
    for k in range(col_data.shape[1]):
      pattern = col_data[:, k]
      partial_pattern = np.zeros_like(pattern)

      ones_indices = np.nonzero(pattern.reshape(-1))
      ones_indices = list(ones_indices)[0]
      num_ones = int(cue_percentages[i] * len(ones_indices))
      remaining_inds = random.sample(range(len(ones_indices)), num_ones)

      partial_pattern[ones_indices[remaining_inds]] = 1

      retrieved = net.retrieve(partial_pattern)
      retrieved_data_matrices[i, k, :] = retrieved.reshape(-1)

      dot_prod = np.dot(pattern, retrieved)
      dot_prod_accuracies[i] += dot_prod

    dot_prod_accuracies[i] /= col_data.shape[1]

  return dot_prod_accuracies, retrieved_data_matrices


# visualize similarities between internal representations during learning and retrieval, relative to original data similarity
def getLearningCovarianceMatrices(net, data):
  y_matrix, out_matrix = net.learn_patterns(data)

  plt.figure()
  plt.imshow(y_matrix)
  plt.colorbar()
  plt.title("Internal Representation Allocation (while learning)")

  cross_data_matrix = data @ data.T

  plt.figure()
  plt.imshow(cross_data_matrix)
  plt.colorbar()
  plt.title("Data Comparison")

  print(np.sum(np.abs(y_matrix), axis=1))

  cross_y_matrix = y_matrix @ y_matrix.T

  plt.figure()
  plt.imshow(cross_y_matrix)
  plt.colorbar()
  plt.title("Internal Representation Comparison")

  print("Internal Representation Stats")
  diag_mean, diag_std = calculate_diagonal_statistics(cross_y_matrix)
  print("Diag mean:" + str(diag_mean), "Diag std: " + str(diag_std))
  offdiag_mean, offdiag_std = calculate_offdiagonal_statistics(cross_y_matrix)
  print("Offdiag mean:" + str(offdiag_mean), "Offdiag std: " + str(offdiag_std))

  cross_out_matrix = out_matrix @ out_matrix.T

  plt.figure()
  plt.imshow(cross_out_matrix)
  plt.colorbar()
  plt.title("Post Recurrence Comparison")

  print("Output Layer Stats")
  diag_mean, diag_std = calculate_diagonal_statistics(cross_out_matrix)
  print("Diag mean:" + str(diag_mean), "Diag std: " + str(diag_std))
  offdiag_mean, offdiag_std = calculate_offdiagonal_statistics(cross_out_matrix)
  print("Offdiag mean:" + str(offdiag_mean), "Offdiag std: " + str(offdiag_std))

  cross_io_matrix = data @ out_matrix.T

  plt.figure()
  plt.imshow(cross_io_matrix)
  plt.colorbar()
  plt.title("Input-Output Comparison (while learning is occurring)")

  print("Input-Output Stats (while learning is occurring)")
  diag_mean, diag_std = calculate_diagonal_statistics(cross_io_matrix)
  print("Diag mean:" + str(diag_mean), "Diag std: " + str(diag_std))
  offdiag_mean, offdiag_std = calculate_offdiagonal_statistics(cross_io_matrix)
  print("Offdiag mean:" + str(offdiag_mean), "Offdiag std: " + str(offdiag_std))

  y_matrix, out_matrix = net.retrieve_patterns(data)

  cross_io_matrix = data @ out_matrix.T

  plt.figure()
  plt.imshow(cross_io_matrix)
  plt.colorbar()
  plt.title("Input-Output Comparison (post-learning)")

  print("Input-Output Stats (after learning is done)")
  diag_mean, diag_std = calculate_diagonal_statistics(cross_io_matrix)
  print("Diag mean:" + str(diag_mean), "Diag std: " + str(diag_std))
  offdiag_mean, offdiag_std = calculate_offdiagonal_statistics(cross_io_matrix)
  print("Offdiag mean:" + str(offdiag_mean), "Offdiag std: " + str(offdiag_std))

  covar_scaling = (((net.input_sparsity*net.input_size))/net.k) * (y_matrix @ y_matrix.T) - (data @ data.T)
  plt.figure()
  plt.imshow( covar_scaling )
  plt.colorbar()
  plt.title("Scaling Between Data Covariance and Hidden Representation Covariance")

  print("Covariance Scaling Statistics")
  diag_mean, diag_std = calculate_diagonal_statistics(covar_scaling)
  print("Diag mean:" + str(diag_mean), "Diag std: " + str(diag_std))
  offdiag_mean, offdiag_std = calculate_offdiagonal_statistics(covar_scaling)
  print("Offdiag mean:" + str(offdiag_mean), "Offdiag std: " + str(offdiag_std))

  print("Overall mean: " + str(np.mean(covar_scaling)), "Overall std: " + str(np.std(covar_scaling)))


  plt.figure()
  plt.imshow(y_matrix)
  plt.colorbar()
  plt.title("Internal Representation Allocation (post learning)")



'''
Generate a partial pattern of a given age
Arguments:  
data            :       a sequence of patterns (num_patterns x input_size)
age             :       the age of the pattern to access
cue_percentage  :       the cue level to use
'''
def get_partial_pattern(data, age, cue_percentage):
  col_data = data.T
  pattern = col_data[:, len(data)-age-1]
  partial_pattern = np.zeros_like(pattern)

  ones_indices = np.nonzero(pattern.reshape(-1))
  ones_indices = list(ones_indices)[0]
  num_ones = int(cue_percentage * len(ones_indices))
  remaining_inds = random.sample(range(len(ones_indices)), num_ones)

  partial_pattern[ones_indices[remaining_inds]] = 1

  return partial_pattern


# Generate a distorted version of the binary pattern 'x' in which 'num_flips' of the 1's are turned to 0's
# and vice versa
def bit_flipped(x, num_flips):
  similar_pattern = np.copy(x)
  ones_indices = np.where(x == 1)[0]
  zero_indices = np.where(x == 0)[0]
  ones_flip_indices = random.sample(list(ones_indices), num_flips)
  zeros_flip_indices = random.sample(list(zero_indices), num_flips)
  similar_pattern[ones_flip_indices] = 0
  similar_pattern[zeros_flip_indices] = 1
  return similar_pattern


# permute the patterns in a dataset of shape num_patterns x input_size
def shuffleData(data):
  shuffle_inds = np.arange(data.shape[0])
  np.random.shuffle(shuffle_inds)
  shuffled_data = data[shuffle_inds.tolist()]
  return shuffled_data

'''
The basic class structure used to generate hierarchical, structured patterns
'''
class NestedTreeNode:
  def __init__(self, pattern_input_size, pattern_sparsity, num_flips):
    self.n = 0    # represents total number of descendants
    self.num_children = 0   # total number of direct children
    self.children = []
    self.child_numbers = []
    self.child_probabilities = np.array([1.])  # first entry in array is probability of creating a new child node for this node

    self.num_flips = num_flips

    self.pattern_input_size = pattern_input_size
    self.pattern_sparsity = pattern_sparsity
    x = generateData(1, pattern_input_size, int(pattern_input_size * pattern_sparsity))[0]
    self.pattern = x  # store pattern for current node

  # add a new child to the current node
  def add_new_child(self):
    assert self.num_children==0
    child = NestedTreeNode(self.pattern_input_size, self.pattern_sparsity, self.num_flips)
    child.pattern = bit_flipped(self.pattern, self.num_flips)
    self.children.append(child)
    self.child_numbers.append(1)
    self.n = 1
    self.num_children = 1
    self.child_probabilities = np.array([0.5, 0.5])

  # run the hierarchical Chinese restaurant process
  def run_restaurant_process(self):
    if self.num_children==0:
      self.add_new_child()
    else:
      chosen_ind = np.random.choice(np.arange(self.num_children + 1), p=self.child_probabilities)
      if chosen_ind==0:   # represents creating a new child node at the end of the children array
        new_child = NestedTreeNode(self.pattern_input_size, self.pattern_sparsity, self.num_flips)
        new_child.pattern = bit_flipped(self.pattern, self.num_flips)
        self.children.append(new_child)
        self.n = self.n + 1
        self.num_children = self.num_children + 1
        self.child_numbers.append(1)
        self.child_probabilities = np.append(self.child_probabilities, 1./self.n) * (self.n / (self.n + 1.))
      else:               # have picked an existing child
        self.child_numbers[chosen_ind-1] = self.child_numbers[chosen_ind-1] + 1
        self.n = self.n + 1
        self.child_probabilities[chosen_ind] = self.child_probabilities[chosen_ind] + 1./self.n
        self.child_probabilities = self.child_probabilities * (self.n / (self.n + 1.))
        chosen_child = self.children[chosen_ind-1]
        chosen_child.run_restaurant_process()


  # generate a hierarchical, structured dataset of patterns 
  def get_clustered_data(self, num_data=None):
    # run Chinese restaurant process to generate hierarchical, clustered data
    if num_data is not None:
      for _ in range(num_data-1):
        self.run_restaurant_process()
      #self.print_tree_structure()

    if len(self.children)==0:
      return np.expand_dims(self.pattern, axis=0)

    # collect all the patterns into a matrix
    data_matrix = None
    for child in self.children:
      if data_matrix is None:
        data_matrix = child.get_clustered_data()
      else:
        data_matrix = np.vstack((data_matrix, child.get_clustered_data()))

    return data_matrix
  
