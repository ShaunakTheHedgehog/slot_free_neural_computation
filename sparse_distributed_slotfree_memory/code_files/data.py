import numpy as np
import random
from utils import bit_flipped

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
