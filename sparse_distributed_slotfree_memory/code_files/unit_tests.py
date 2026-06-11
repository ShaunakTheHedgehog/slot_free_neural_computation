'''
Unit tests for the sparse, distributed, slot-free memory code.

Run from within the `code_files/` directory (so that the sibling modules import correctly) with:

    python -m unittest unit_tests -v

or simply:

    python unit_tests.py

The tests vet the data-generation utilities (data.py), the general helper functions (utils.py),
and the core K-winner MHN model (kwinnernet.py).
'''

import unittest
import random

import numpy as np

import globals
from utils import (
    softmax,
    sparsify,
    one_winner_take_all,
    hard_k_winner_take_all,
    random_k_winner_take_all,
    bit_flipped,
    shuffleData,
    calculate_diagonal_statistics,
    calculate_offdiagonal_statistics,
)
from data import (
    generateData,
    generate_clustered_data,
    generate_random_dataset,
    generate_correlated_dataset,
    generate_tree_dataset,
    NestedTreeNode,
)
from kwinnernet import KWinnerNet


def _seed(seed=0):
    '''Seed both RNGs used throughout the codebase so randomized tests are reproducible.'''
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------
class TestUtils(unittest.TestCase):

    def test_softmax_is_a_distribution(self):
        x = np.array([1.0, 2.0, 3.0, -1.0])
        out = softmax(x, slope_param=1.0)
        self.assertAlmostEqual(np.sum(out), 1.0, places=10)
        self.assertTrue(np.all(out > 0))
        # larger inputs should receive larger probability mass
        self.assertEqual(np.argmax(out), np.argmax(x))

    def test_softmax_slope_sharpens(self):
        x = np.array([1.0, 2.0, 3.0])
        sharp = softmax(x, slope_param=10.0)
        flat = softmax(x, slope_param=0.1)
        # a higher slope (inverse temperature) concentrates mass on the max entry
        self.assertGreater(sharp[np.argmax(x)], flat[np.argmax(x)])

    def test_sparsify_keeps_largest_entries(self):
        x = np.array([0.1, 0.9, 0.4, 0.8, 0.2])
        out = sparsify(x, sparsity=0.4)  # keep int(0.4 * 5) = 2 entries
        self.assertEqual(int(out.sum()), 2)
        # the two largest entries (indices 1 and 3) should be the active ones
        self.assertEqual(set(np.nonzero(out)[0].tolist()), {1, 3})

    def test_one_winner_take_all(self):
        x = np.array([0.3, 0.1, 0.7, 0.2])
        out = one_winner_take_all(x)
        self.assertEqual(int(out.sum()), 1)
        self.assertEqual(np.argmax(out), 2)

    def test_hard_k_winner_take_all(self):
        x = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        out = hard_k_winner_take_all(x, k=3)
        self.assertEqual(int(out.sum()), 3)
        # top-3 entries are at indices 0, 2, 4
        self.assertEqual(set(np.nonzero(out)[0].tolist()), {0, 2, 4})

    def test_random_k_winner_take_all_count(self):
        _seed(0)
        x = np.zeros(50)
        out = random_k_winner_take_all(x, k=7)
        self.assertEqual(int(out.sum()), 7)
        self.assertTrue(set(np.unique(out).tolist()).issubset({0.0, 1.0}))

    def test_bit_flipped_preserves_sparsity(self):
        _seed(0)
        x = np.zeros(100)
        x[random.sample(range(100), 10)] = 1.0
        num_flips = 3
        flipped = bit_flipped(x, num_flips)
        # number of active bits is preserved (num_flips ones -> zeros, num_flips zeros -> ones)
        self.assertEqual(int(flipped.sum()), int(x.sum()))
        # exactly 2 * num_flips positions differ from the original
        self.assertEqual(int(np.sum(flipped != x)), 2 * num_flips)
        # the original pattern must not be mutated in place
        self.assertEqual(int(x.sum()), 10)

    def test_shuffle_data_preserves_rows(self):
        _seed(0)
        data = generateData(50, 30, 5)
        shuffled = shuffleData(data)
        self.assertEqual(shuffled.shape, data.shape)
        # the same set of rows should be present, just reordered
        orig = sorted(tuple(row) for row in data.tolist())
        new = sorted(tuple(row) for row in shuffled.tolist())
        self.assertEqual(orig, new)

    def test_diagonal_statistics(self):
        m = np.array([[2.0, 9.0], [9.0, 4.0]])
        mean, std = calculate_diagonal_statistics(m)
        self.assertAlmostEqual(mean, 3.0)        # (2 + 4) / 2
        self.assertAlmostEqual(std, 1.0)         # std of [2, 4]

    def test_offdiagonal_statistics_mean(self):
        m = np.array([[0.0, 5.0], [7.0, 0.0]])
        mean, _ = calculate_offdiagonal_statistics(m)
        self.assertAlmostEqual(mean, 6.0)        # (5 + 7) / 2


# ---------------------------------------------------------------------------
# data.py
# ---------------------------------------------------------------------------
class TestGenerateData(unittest.TestCase):

    def test_shape_and_sparsity(self):
        _seed(0)
        data = generateData(num_examples=40, length=120, num_active=15)
        self.assertEqual(data.shape, (40, 120))
        self.assertTrue(np.all(data.sum(axis=1) == 15))
        self.assertTrue(set(np.unique(data).tolist()).issubset({0.0, 1.0}))


class TestGenerateClusteredData(unittest.TestCase):

    def test_shape_and_sparsity(self):
        _seed(0)
        data = generate_clustered_data(num_categories=5, num_examples_per_category=20,
                                       num_flips=3, length=100, num_active=10)
        self.assertEqual(data.shape, (5 * 20, 100))
        self.assertTrue(np.all(data.sum(axis=1) == 10))
        self.assertTrue(set(np.unique(data).tolist()).issubset({0.0, 1.0}))

    def test_descendants_close_to_supplied_prototypes(self):
        _seed(0)
        length, num_active, num_flips = 200, 20, 4
        prototypes = generateData(3, length, num_active)
        data = generate_clustered_data(num_categories=3, num_examples_per_category=50,
                                       num_flips=num_flips, length=length, num_active=num_active,
                                       prototypes=prototypes)
        # every descendant must lie within Hamming distance 2 * num_flips of some prototype,
        # i.e. its overlap with that prototype is at least num_active - num_flips
        best_overlap = (data @ prototypes.T).max(axis=1)
        self.assertTrue(np.all(best_overlap >= num_active - num_flips))

    def test_clustering_structure(self):
        # within-category similarity should clearly exceed across-category similarity
        _seed(1)
        length, num_active, num_flips = 300, 30, 3
        n_per = 40
        data = generate_clustered_data(num_categories=4, num_examples_per_category=n_per,
                                       num_flips=num_flips, length=length, num_active=num_active)
        # data is shuffled; recover category identity via nearest-prototype is not possible here,
        # so instead check that the pairwise-overlap distribution is bimodal: the top overlaps
        # (same category) are much higher than the median (cross category)
        overlaps = data @ data.T
        off_diag = overlaps[~np.eye(len(data), dtype=bool)]
        high = np.percentile(off_diag, 95)
        median = np.percentile(off_diag, 50)
        self.assertGreater(high, median + num_active / 2)


class TestDatasetBuilders(unittest.TestCase):

    EXPECTED = (globals.NUM_BURN_IN, globals.NUM_EVAL, globals.NUM_EVAL)

    def _check_splits(self, burn_in, train, pseudo, length, num_active):
        self.assertEqual(burn_in.shape, (globals.NUM_BURN_IN, length))
        self.assertEqual(train.shape, (globals.NUM_EVAL, length))
        self.assertEqual(pseudo.shape, (globals.NUM_EVAL, length))
        for arr in (burn_in, train, pseudo):
            self.assertTrue(np.all(arr.sum(axis=1) == num_active))
            self.assertTrue(set(np.unique(arr).tolist()).issubset({0.0, 1.0}))

    def test_globals_are_consistent(self):
        # the burn-in + recent split should account for the total pattern budget
        self.assertEqual(globals.NUM_BURN_IN + globals.NUM_EVAL, globals.TOT_NUM_PATTERNS)

    def test_random_dataset(self):
        _seed(0)
        length, num_active = 200, 20
        burn_in, train, pseudo = generate_random_dataset(length, num_active)
        self._check_splits(burn_in, train, pseudo, length, num_active)

    def test_correlated_dataset_shapes(self):
        _seed(0)
        length, num_active = 200, 20
        burn_in, train, pseudo = generate_correlated_dataset(length, num_active, num_flips=3,
                                                             num_categories=10)
        self._check_splits(burn_in, train, pseudo, length, num_active)

    def test_correlated_dataset_shares_categories(self):
        # burn-in and recent-training patterns come from the SAME categories, whereas the
        # pseudo-patterns come from other categories -> burn-in should match train far better
        _seed(1)
        length, num_active, num_flips = 300, 30, 3
        burn_in, train, pseudo = generate_correlated_dataset(length, num_active, num_flips,
                                                             num_categories=10)
        sample = burn_in[:200]
        best_with_train = np.mean((sample @ train.T).max(axis=1))
        best_with_pseudo = np.mean((sample @ pseudo.T).max(axis=1))
        self.assertGreater(best_with_train, best_with_pseudo + num_active / 2)

    def test_tree_dataset(self):
        _seed(0)
        length, num_active = 200, 20
        burn_in, train, pseudo = generate_tree_dataset(length, num_active, num_flips=5)
        self._check_splits(burn_in, train, pseudo, length, num_active)


class TestNestedTreeNode(unittest.TestCase):

    def test_root_pattern_sparsity(self):
        _seed(0)
        length, sparsity = 200, 0.1
        tree = NestedTreeNode(pattern_input_size=length, pattern_sparsity=sparsity, num_flips=5)
        self.assertEqual(int(tree.pattern.sum()), int(length * sparsity))

    def test_clustered_data_shape_and_sparsity(self):
        _seed(0)
        length, sparsity = 200, 0.1
        tree = NestedTreeNode(pattern_input_size=length, pattern_sparsity=sparsity, num_flips=5)
        data = tree.get_clustered_data(num_data=500)
        self.assertEqual(data.shape[1], length)
        self.assertGreater(data.shape[0], 0)
        self.assertTrue(np.all(data.sum(axis=1) == int(length * sparsity)))

    def test_child_probabilities_normalized(self):
        _seed(0)
        tree = NestedTreeNode(pattern_input_size=100, pattern_sparsity=0.1, num_flips=3)
        for _ in range(50):
            tree.run_restaurant_process()
        # the child-probability vector (incl. the new-child slot) must remain a distribution
        self.assertAlmostEqual(float(np.sum(tree.child_probabilities)), 1.0, places=8)


# ---------------------------------------------------------------------------
# kwinnernet.py
# ---------------------------------------------------------------------------
class TestKWinnerNet(unittest.TestCase):

    def _make_net(self, input_size=100, hidden_size=200, num_active=10,
                  fan_in_ratio=1.0, k=5, eta=1.0, nonlinearity_type='hard_k'):
        return KWinnerNet(input_size=input_size, hidden_size=hidden_size,
                          input_sparsity=1.0 * num_active / input_size,
                          fan_in_ratio=fan_in_ratio, k=k, eta=eta,
                          nonlinearity_type=nonlinearity_type)

    def test_invalid_nonlinearity_rejected(self):
        with self.assertRaises(AssertionError):
            self._make_net(nonlinearity_type='not_a_rule')

    def test_architecture_fan_in(self):
        _seed(0)
        net = self._make_net(input_size=100, hidden_size=50, fan_in_ratio=0.2)
        # every hidden unit receives exactly fan_in_ratio * input_size connections
        self.assertTrue(np.all(net.W_xy_architecture.sum(axis=1) == 20))
        # the weight matrices are masked by the connectivity architecture (and its transpose)
        self.assertTrue(np.all((net.W_xy != 0) <= (net.W_xy_architecture != 0)))
        self.assertTrue(np.all((net.W_yx != 0) <= (net.W_xy_architecture.T != 0)))

    def test_forward_hidden_has_k_winners(self):
        _seed(0)
        net = self._make_net(k=5)
        x = generateData(1, 100, 10)[0].reshape(-1, 1)
        net.forward(x, phase='learning')
        self.assertEqual(int(net.y.sum()), 5)

    def test_retrieval_output_sparsity(self):
        _seed(0)
        num_active = 10
        net = self._make_net(num_active=num_active)
        x = generateData(1, 100, num_active)[0].reshape(-1, 1)
        out = net.retrieve(x)
        self.assertEqual(int(out.sum()), num_active)

    def test_recently_learned_pattern_reconstructs(self):
        # after learning a short sequence, the most-recent pattern should be recovered exactly
        # from a full cue (the K-winner MHN binds it strongly in its fresh weights)
        _seed(0)
        num_active = 10
        net = self._make_net(num_active=num_active, eta=1.0)
        data = generateData(5, 100, num_active)
        net.learn_patterns(data)
        out = net.retrieve(data[-1].reshape(-1, 1)).reshape(-1)
        overlap = int(np.dot(out, data[-1]))
        self.assertEqual(overlap, num_active)

    def test_retrieval_does_not_change_weights(self):
        _seed(0)
        net = self._make_net()
        data = generateData(3, 100, 10)
        net.learn_patterns(data)
        before = net.W_xy.copy()
        net.retrieve_patterns(data)
        self.assertTrue(np.array_equal(before, net.W_xy))


if __name__ == '__main__':
    unittest.main(verbosity=2)
