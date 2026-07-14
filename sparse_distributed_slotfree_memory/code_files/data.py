import numpy as np
import random
from utils import bit_flipped, shuffleData
from globals import NUM_BURN_IN, NUM_EVAL, TOT_NUM_PATTERNS

'''
Generate a dataset of binary patterns, all with the same sparsity level
Arguments:
num_examples    :     number of patterns to generate  
length          :     total length of each binary pattern
num_active      :     number of 1-bits in each pattern (i.e. effectively the sparsity level)

Returns:
data            :     a 2D array of binary patterns (with shape num_examples x length)
'''
def generate_data(num_examples, length, num_active):
  data = np.zeros((num_examples, length))
  for i in range(num_examples):
    data_vec = np.zeros(length)
    rand_indices = random.sample(range(length), num_active)
    data_vec[rand_indices] = 1.
    data[i] = data_vec
  return data


'''
Generate a dataset of correlated (clustered) binary patterns.
First, 'num_categories' iid random binary prototype/parent patterns are drawn (each of length 'length'
with 'num_active' 1-bits). Then, for each prototype, 'num_examples_per_category' descendant patterns are
generated, each a randomly bit-flipped version of that prototype ('num_flips' of the 1-bits flipped to 0,
and 'num_flips' of the 0-bits flipped to 1). All descendant patterns are pooled together and shuffled.

Arguments:
num_categories              :     number of prototype/parent patterns (i.e. categories)
num_examples_per_category   :     number of descendant patterns generated per prototype
num_flips                   :     number of bit flips applied to a prototype to produce each descendant
length                      :     total length of each binary pattern
num_active                  :     number of 1-bits in each pattern (i.e. the sparsity level)
prototypes                  :     optional array of pre-generated prototypes (shape num_categories x length).
                                  If None (default), the prototypes are drawn iid as described above. Supplying
                                  prototypes allows multiple datasets to share the same underlying categories.

Returns:
data        :     a shuffled 2D array of binary patterns (with shape (num_categories * num_examples_per_category) x length)
'''
def generate_clustered_data(num_categories, num_examples_per_category, num_flips, length, num_active, prototypes=None):
    if prototypes is None:
        # draw the iid random binary prototype/parent patterns
        prototypes = generate_data(num_categories, length, num_active)

    data = np.zeros((num_categories * num_examples_per_category, length))
    idx = 0
    for c in range(num_categories):
        for _ in range(num_examples_per_category):
            # each descendant is a randomly bit-flipped version of its parent prototype
            data[idx] = bit_flipped(prototypes[c], num_flips)
            idx += 1

    # shuffle the ordering of all the generated patterns
    return shuffleData(data)


'''
Build a dataset of random, iid (unstructured) binary patterns for training and testing the K-winner MHN.
The training set is split into a 'burn-in' phase of NUM_TRAIN patterns (presented first, to bring the
network into steady state) followed by the NUM_TEST most recent patterns that are actually evaluated.
A further NUM_TEST untrained pseudo-patterns are generated; these are never learned by the network and
serve as a baseline. All patterns share the same length and number of 1-bits.

Arguments:
length      :     total length of each binary pattern
num_active  :     number of 1-bits in each pattern (i.e. the sparsity level)

Returns:
burn_in_data    :     NUM_TRAIN burn-in patterns (shape NUM_TRAIN x length)
train_data      :     NUM_TEST most-recent training patterns (shape NUM_TEST x length)
pseudo_data     :     NUM_TEST untrained pseudo-patterns (shape NUM_TEST x length)
'''
def generate_random_dataset(length, num_active, num_burn_in=NUM_BURN_IN, num_eval=NUM_EVAL):
    burn_in_data = generate_data(num_burn_in, length, num_active)
    train_data = generate_data(num_eval, length, num_active)
    pseudo_data = generate_data(num_eval, length, num_active)
    return burn_in_data, train_data, pseudo_data


'''
Build a dataset of correlated (clustered) binary patterns for training and testing the K-winner MHN.
The NUM_TRAIN burn-in patterns consist of (NUM_TRAIN // num_categories) iid examples from each of
'num_categories' categories, and the NUM_TEST most-recent training patterns consist of
(NUM_TEST // num_categories) iid examples from these SAME categories. The NUM_TEST untrained
pseudo-patterns are generated from a separate set of 'num_categories' randomly generated categories,
so that they share the same clustered statistics but are never seen by the network.

By default num_categories=10, which (with NUM_TRAIN=3000, NUM_TEST=1000) yields 300 burn-in examples
and 100 recent examples per category, and 100 pseudo examples per (other) category.

Arguments:
length          :     total length of each binary pattern
num_active      :     number of 1-bits in each pattern (i.e. the sparsity level)
num_flips       :     number of bit flips used to derive a descendant from its category prototype
num_categories  :     number of categories (prototypes) underlying the data (default: 10)

Returns:
burn_in_data    :     NUM_TRAIN burn-in patterns (shape NUM_TRAIN x length)
train_data      :     NUM_TEST most-recent training patterns (shape NUM_TEST x length)
pseudo_data     :     NUM_TEST untrained pseudo-patterns (shape NUM_TEST x length)
'''
def generate_correlated_dataset(length, num_active, num_flips, num_categories=10, num_burn_in=NUM_BURN_IN, num_eval=NUM_EVAL,
                                pseudo_from_same_categories=True):
    # the burn-in and recent training patterns are drawn from the SAME set of categories
    train_prototypes = generate_data(num_categories, length, num_active)
    burn_in_data = generate_clustered_data(num_categories, num_burn_in // num_categories, num_flips,
                                           length, num_active, prototypes=train_prototypes)
    train_data = generate_clustered_data(num_categories, num_eval // num_categories, num_flips,
                                         length, num_active, prototypes=train_prototypes)

    # the pseudo-patterns come from a separate, never-seen set of categories
    if pseudo_from_same_categories:
        pseudo_prototypes = train_prototypes
    else:
        pseudo_prototypes = generate_data(num_categories, length, num_active)
    pseudo_data = generate_clustered_data(num_categories, num_eval // num_categories, num_flips,
                                          length, num_active, prototypes=pseudo_prototypes)

    return burn_in_data, train_data, pseudo_data


'''
Build a dataset of hierarchically structured (NestedTreeNode) binary patterns for training and testing
the K-winner MHN. A single tree is grown via the hierarchical Chinese-restaurant process and its leaf
patterns are pooled, shuffled, and split into NUM_TRAIN burn-in patterns, NUM_TEST most-recent training
patterns, and NUM_TEST untrained pseudo-patterns. Because the tree yields roughly half as many leaf
patterns as the number of restaurant-process steps requested, the tree is over-grown and then truncated
to the exact counts needed.

Arguments:
length      :     total length of each binary pattern
num_active  :     number of 1-bits in each pattern (i.e. the sparsity level)
num_flips   :     number of bit flips applied at each level of the tree-generating process

Returns:
burn_in_data    :     NUM_TRAIN burn-in patterns (shape NUM_TRAIN x length)
train_data      :     NUM_TEST most-recent training patterns (shape NUM_TEST x length)
pseudo_data     :     NUM_TEST untrained pseudo-patterns (shape NUM_TEST x length)
'''
def generate_tree_dataset(length, num_active, num_flips, num_burn_in=NUM_BURN_IN, num_eval=NUM_EVAL):
    needed = num_burn_in + 2 * num_eval
    sparsity = 1. * num_active / length

    # over-grow the tree, since it produces roughly num_data/2 leaf patterns
    tree = NestedTreeNode(pattern_input_size=length, pattern_sparsity=sparsity, num_flips=num_flips)
    full_data = tree.get_clustered_data(num_data=3 * needed)
    assert full_data.shape[0] >= needed, \
        f'tree produced only {full_data.shape[0]} patterns, need {needed}'

    full_data = shuffleData(full_data)
    burn_in_data = full_data[:num_burn_in]
    train_data = full_data[num_burn_in:num_burn_in + num_eval]
    pseudo_data = full_data[num_burn_in + num_eval:num_burn_in + 2 * num_eval]

    return burn_in_data, train_data, pseudo_data


def generate_specific_dataset(length, num_active, data_type='random', num_flips=None, num_categories=10, num_burn_in=NUM_BURN_IN, num_eval=NUM_EVAL):
    if data_type == 'random':
        return generate_random_dataset(length, num_active, num_burn_in=num_burn_in, num_eval=num_eval)
    elif data_type == 'correlated':
        assert num_flips is not None
        return generate_correlated_dataset(length, num_active, num_flips, num_categories=num_categories, num_burn_in=num_burn_in, num_eval=num_eval)
    elif data_type == 'tree':
        assert num_flips is not None
        return generate_tree_dataset(length, num_active, num_flips, num_burn_in=num_burn_in, num_eval=num_eval)
    else:
        raise ValueError(f'Invalid data type: {data_type}')


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
    x = generate_data(1, pattern_input_size, int(pattern_input_size * pattern_sparsity))[0]
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
