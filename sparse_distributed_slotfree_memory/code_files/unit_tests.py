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
import copy
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
    shuffle_data,
    calculate_diagonal_statistics,
    calculate_offdiagonal_statistics,
    auc_trapezoid,
)
from data import (
    generate_data,
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
        data = generate_data(50, 30, 5)
        shuffled = shuffle_data(data)
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
        data = generate_data(num_examples=40, length=120, num_active=15)
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
        prototypes = generate_data(3, length, num_active)
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
        # with pseudo_from_same_categories=False the pseudo-patterns come from a separate set of
        # categories, so burn-in patterns should match the training patterns far better than those
        _seed(1)
        length, num_active, num_flips = 300, 30, 3
        burn_in, train, pseudo = generate_correlated_dataset(length, num_active, num_flips,
                                                             num_categories=10,
                                                             pseudo_from_same_categories=False)
        sample = burn_in[:200]
        best_with_train = np.mean((sample @ train.T).max(axis=1))
        best_with_pseudo = np.mean((sample @ pseudo.T).max(axis=1))
        self.assertGreater(best_with_train, best_with_pseudo + num_active / 2)

    def test_correlated_dataset_pseudo_from_same_categories(self):
        # with the default (same categories), pseudo-patterns should match the training patterns
        # about as well as the training patterns match each other
        _seed(1)
        length, num_active, num_flips = 300, 30, 3
        burn_in, train, pseudo = generate_correlated_dataset(length, num_active, num_flips,
                                                             num_categories=10,
                                                             pseudo_from_same_categories=True)
        sample = burn_in[:200]
        best_with_train = np.mean((sample @ train.T).max(axis=1))
        best_with_pseudo = np.mean((sample @ pseudo.T).max(axis=1))
        self.assertLess(abs(best_with_train - best_with_pseudo), num_active / 2)

    def test_correlated_dataset_order_match(self):
        # order_match pairs each real pattern with a pseudo-pattern from the SAME category at the
        # same position, so every matched pair must be far more similar than an arbitrary pair
        _seed(2)
        length, num_active, num_flips, num_categories = 300, 30, 3, 10
        num_burn_in, num_eval = 100, 50
        _, train, pseudo = generate_correlated_dataset(length, num_active, num_flips,
                                                       num_categories=num_categories,
                                                       num_burn_in=num_burn_in, num_eval=num_eval,
                                                       order_match=True)
        matched = np.sum(train * pseudo, axis=1)
        # a same-category pair differs by at most 2*num_flips bits on each side of the prototype
        self.assertTrue(np.all(matched >= num_active - 4 * num_flips))
        # and matched pairs must clearly beat the average unmatched pair
        cross = train @ pseudo.T
        unmatched_mean = cross[~np.eye(num_eval, dtype=bool)].mean()
        self.assertGreater(matched.mean(), unmatched_mean + num_active / 2)

    def test_order_match_requires_same_categories(self):
        with self.assertRaises(AssertionError):
            generate_correlated_dataset(200, 20, 2, num_categories=5, num_burn_in=50, num_eval=10,
                                        pseudo_from_same_categories=False, order_match=True)

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
'''
The original full-matrix form of the K-winner MHN weight update, kept here purely as a
reference implementation. KWinnerNet.__adjust_weights was optimized to touch only the k
winning rows of W_xy (and columns of W_yx), since every other row is multiplied by y=0;
the tests below guard that the optimization stays bit-for-bit equivalent to this rule.
'''
def _reference_adjust_weights(W_xy, W_yx, arch, y, x, eta):
    W_xy_new = W_xy + eta * (np.outer(y, x) - W_xy * y) * arch
    W_yx_new = W_yx + eta * (np.outer(x, y) - W_yx * y.T) * arch.T
    return W_xy_new, W_yx_new


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
        x = generate_data(1, 100, 10)[0].reshape(-1, 1)
        net.forward(x, phase='learning')
        self.assertEqual(int(net.y.sum()), 5)

    def test_retrieval_output_sparsity(self):
        _seed(0)
        num_active = 10
        net = self._make_net(num_active=num_active)
        x = generate_data(1, 100, num_active)[0].reshape(-1, 1)
        out = net.retrieve(x)
        self.assertEqual(int(out.sum()), num_active)

    def test_recently_learned_pattern_reconstructs(self):
        # after learning a short sequence, the most-recent pattern should be recovered exactly
        # from a full cue (the K-winner MHN binds it strongly in its fresh weights)
        _seed(0)
        num_active = 10
        net = self._make_net(num_active=num_active, eta=1.0)
        data = generate_data(5, 100, num_active)
        net.learn_patterns(data)
        out = net.retrieve(data[-1].reshape(-1, 1)).reshape(-1)
        overlap = int(np.dot(out, data[-1]))
        self.assertEqual(overlap, num_active)

    def test_retrieval_does_not_change_weights(self):
        _seed(0)
        net = self._make_net()
        data = generate_data(3, 100, 10)
        net.learn_patterns(data)
        before = net.W_xy.copy()
        net.retrieve_patterns(data)
        self.assertTrue(np.array_equal(before, net.W_xy))

    def test_sparse_weight_update_matches_full_update(self):
        # the optimized (winners-only) weight update must reproduce the original full-matrix
        # rule exactly, across a range of network shapes, fan-in ratios and learning rates
        for (n_i, n_h, num_active, f, k, eta) in [(100, 200, 10, 1.0, 5, 1.0),
                                                  (100, 200, 10, 0.3, 7, 0.35),
                                                  (60, 40, 6, 0.5, 3, 0.7),
                                                  (80, 50, 8, 1.0, 1, 1.0)]:
            _seed(0)
            net = self._make_net(input_size=n_i, hidden_size=n_h, num_active=num_active,
                                 fan_in_ratio=f, k=k, eta=eta)
            for _ in range(5):
                x = generate_data(1, n_i, num_active)[0].reshape(-1, 1)
                # a retrieval pass sets the hidden state without touching the weights
                net.forward(x, phase='retrieval')
                expected_xy, expected_yx = _reference_adjust_weights(
                    net.W_xy.copy(), net.W_yx.copy(), net.W_xy_architecture, net.y, x, net.eta)
                net._KWinnerNet__adjust_weights(x)
                self.assertTrue(np.array_equal(net.W_xy, expected_xy))
                self.assertTrue(np.array_equal(net.W_yx, expected_yx))

    def test_sparse_weight_update_matches_for_arbitrary_hidden_state(self):
        # covers the 'random' k-winner rule, where the winners are not the top-k logits
        _seed(0)
        n_i, n_h, k = 100, 200, 5
        net = self._make_net(input_size=n_i, hidden_size=n_h, num_active=10,
                             fan_in_ratio=0.4, k=k, eta=0.5)
        x = generate_data(1, n_i, 10)[0].reshape(-1, 1)
        y = np.zeros((n_h, 1))
        y[np.random.choice(n_h, k, replace=False)] = 1.
        net.y = y
        expected_xy, expected_yx = _reference_adjust_weights(
            net.W_xy.copy(), net.W_yx.copy(), net.W_xy_architecture, net.y, x, net.eta)
        net._KWinnerNet__adjust_weights(x)
        self.assertTrue(np.array_equal(net.W_xy, expected_xy))
        self.assertTrue(np.array_equal(net.W_yx, expected_yx))

    def test_sparse_weight_update_matches_over_learning_sequence(self):
        # errors could accumulate across sequential updates, so check a whole learning run
        _seed(0)
        n_i, num_active = 100, 10
        net_a = self._make_net(input_size=n_i, hidden_size=200, num_active=num_active,
                               fan_in_ratio=0.5, k=5, eta=0.3)
        net_b = copy.deepcopy(net_a)
        data = generate_data(50, n_i, num_active)

        for j in range(data.shape[0]):
            x = data[j].reshape(-1, 1)
            net_a.forward(x, phase='learning')      # optimized winners-only update
            # net_b picks the same top-k winners, then applies the reference update by hand
            net_b.forward(x, phase='retrieval')
            net_b.W_xy, net_b.W_yx = _reference_adjust_weights(
                net_b.W_xy, net_b.W_yx, net_b.W_xy_architecture, net_b.y, x, net_b.eta)
            self.assertTrue(np.array_equal(net_a.W_xy, net_b.W_xy), f'W_xy diverged at pattern {j}')
            self.assertTrue(np.array_equal(net_a.W_yx, net_b.W_yx), f'W_yx diverged at pattern {j}')


# ---------------------------------------------------------------------------
# AUC and the raw-data analysis pipeline (utils.py, retrieval_analyses.py)
# ---------------------------------------------------------------------------
class TestAUC(unittest.TestCase):

    def test_auc_matches_mann_whitney(self):
        # AUC computed from histograms must equal the Mann-Whitney U statistic normalized by
        # n1*n2, which is the standard definition (and credits ties with 0.5)
        from scipy.stats import mannwhitneyu
        rng = np.random.default_rng(0)
        for _ in range(5):
            real = rng.integers(0, 101, size=20)
            pseudo = rng.integers(0, 101, size=20)
            auc = auc_trapezoid(np.bincount(real, minlength=101), np.bincount(pseudo, minlength=101))
            u = mannwhitneyu(real, pseudo, alternative='greater').statistic
            self.assertAlmostEqual(auc, u / (len(real) * len(pseudo)), places=10)

    def test_auc_edge_cases(self):
        from retrieval_analyses import aucs_across_ages
        num_active = 10
        high = np.full((5, 3), num_active)
        low = np.zeros((5, 3), dtype=int)
        same = np.tile(np.array([[4, 5, 6]]), (5, 1))
        # perfectly separated, identical, and perfectly reversed distributions
        self.assertTrue(np.allclose(aucs_across_ages(high, low, num_active), 1.0))
        self.assertTrue(np.allclose(aucs_across_ages(same, same, num_active), 0.5))
        self.assertTrue(np.allclose(aucs_across_ages(low, high, num_active), 0.0))

    def test_auc_curves_shape(self):
        from retrieval_analyses import auc_curves
        _seed(0)
        num_samples, num_runs, num_mems, num_active = 3, 6, 8, 10
        real = np.random.randint(0, num_active + 1, (num_samples, num_runs, num_mems))
        pseudo = np.random.randint(0, num_active + 1, (num_samples, num_runs, num_mems))
        out = auc_curves(real, pseudo, num_active)
        self.assertEqual(out.shape, (num_samples, num_mems))
        self.assertTrue(np.all((out >= 0.) & (out <= 1.)))


class TestRawAnalysisPipeline(unittest.TestCase):

    def test_dprime_and_rawdiff_from_raw_match_get_match_probabilities(self):
        # the two-stage pipeline (simulate -> save raw -> derive curves) must reproduce the d' and
        # raw differences that the original single-stage routine computes
        import kwinner_mhn_comparison as K
        from retrieval_analyses import dprime_curves, rawdiff_curves

        runset = (200, 400, 20, 0.2, 5, 0.5)
        num_active = runset[2]

        _seed(3)
        _, _, dprime_ref, rawdiff_ref, _ = K.get_match_probabilities(
            runset, 6, 12, 1.0, num_burn_in=40, data_type='random')

        _seed(3)
        real, pseudo, _ = K.simulate_trials(runset, 6, 12, 1.0, num_burn_in=40, data_type='random')

        dprime_raw = dprime_curves(real[None, :, :], pseudo[None, :, :], num_active)[0]
        rawdiff_raw = rawdiff_curves(real[None, :, :], pseudo[None, :, :], num_active)[0]

        self.assertTrue(np.allclose(dprime_ref, dprime_raw, equal_nan=True))
        self.assertTrue(np.allclose(rawdiff_ref, rawdiff_raw))

    def test_raw_collection_shapes_and_metadata(self):
        import kwinner_mhn_comparison as K
        from retrieval_analyses import compute_curves

        _seed(0)
        runset1, runset2 = (200, 400, 20, 0.2, 5, 0.5), (200, 100, 20, 1.0, 1, 1.0)
        num_samples, num_runs, num_mems = 2, 3, 10
        raw = K.run_raw_collection(runset1, runset2, num_mems, num_samples, num_runs, 1.0,
                                   num_burn_in=30, data_type='correlated', num_flips=2,
                                   num_categories=5, uniform_baseline=True, save_data=False)

        for key in ('kwin_real', 'kwin_pseudo', 'kwin_unif', 'mhn_real', 'mhn_pseudo', 'mhn_unif'):
            self.assertEqual(raw[key].shape, (num_samples, num_runs, num_mems))
            # overlaps are integer bit-match counts, so they must lie in 0...num_active
            self.assertTrue(np.all(raw[key] <= runset1[2]))
        self.assertEqual(raw['meta']['num_active'], runset1[2])

        curves = compute_curves(raw)
        for key in ('kwinner_dprimes', 'mhn_dprimes', 'kwinner_aucs', 'mhn_aucs',
                    'kwinner_unif_aucs', 'kwin_out_accs'):
            self.assertEqual(curves[key].shape, (num_samples, num_mems))

    def test_raw_collection_rejects_mismatched_sparsity(self):
        import kwinner_mhn_comparison as K
        _seed(0)
        with self.assertRaises(AssertionError):
            K.run_raw_collection((200, 400, 20, 0.2, 5, 0.5), (200, 100, 15, 1.0, 1, 1.0),
                                 10, 1, 1, 1.0, num_burn_in=20, save_data=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
