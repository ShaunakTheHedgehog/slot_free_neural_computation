import numpy as np
import random
from utils import sparsify, hard_k_winner_take_all, softmax, get_partial_pattern, random_k_winner_take_all

'''
Main model of interest: the K-winner MHN
'''
class KWinnerNet:

    '''
    Initialization arguments:
    input_size          :       size of input layer (n_i)
    hidden_size         :       size of hidden layer (n_h)
    fan_in_ratio        :       proportion of input units connecting to a hidden unit (f)
    k                   :       number of winners at hidden layer (k)
    eta                 :       exponentially averaged learning rate parameter for weights (epsilon)
    nonlinearity_type   :       whether the nonlinearity is a "top-k" rule or a "pick k random units" rule

    '''
    def __init__(self, input_size, hidden_size, input_sparsity, fan_in_ratio, k, eta=1.0, nonlinearity_type='hard_k'):
        assert nonlinearity_type in ['hard_k', 'random']
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fan_in_ratio = fan_in_ratio
        self.num_sending_neurons = int(fan_in_ratio * input_size)

        self.input_sparsity = input_sparsity

        self.W_xy_architecture = self.__generateMatrixArchitecture()
        self.W_xy = np.random.rand(hidden_size, input_size) * self.W_xy_architecture

        self.W_yx = np.random.rand(input_size,
                                   hidden_size) * self.W_xy_architecture.T  

        self.logits = np.zeros((hidden_size, 1))
        self.y = np.zeros((hidden_size, 1))

        self.nonlinearity_type = nonlinearity_type

        self.eta = eta
        self.k = k


    # generate the binary fan-in matrix that says which input-to-hidden connections exist
    def __generateMatrixArchitecture(self):
        mat = np.zeros((self.hidden_size, self.input_size))

        for i in range(self.hidden_size):
            # for each hidden neuron, randomly sample f * n_i input neurons to project to it
            sending_neuron_pattern = np.zeros(self.input_size)
            active_indices = random.sample(range(self.input_size), self.num_sending_neurons)
            sending_neuron_pattern[active_indices] = 1.
            mat[i] = sending_neuron_pattern

        return mat

    # update rule for the weights, which is symmetric for W_xy and W_yx
    # only the incoming weights to the k winning hidden units get updated under this rule
    def __adjust_weights(self, x):
        self.W_xy = self.W_xy + self.eta * (np.outer(self.y, x) - self.W_xy * self.y) * self.W_xy_architecture
        self.W_yx = self.W_yx + self.eta * (np.outer(x, self.y) - self.W_yx * self.y.T) * self.W_xy_architecture.T

    '''
    Pass a pattern through the K-winner MHN, either to learn it or retrieve a completed pattern
    Arguments:
    x       :       the input binary pattern of size n_i    
    phase   :       whether the pattern is being learned (i.e. weights updated) or simply retrieved (with weights frozen)
    '''
    def forward(self, x, phase="learning"):
        assert phase in ['learning', 'retrieval']

        self.logits = self.W_xy @ x

        if phase == 'learning':
            # adjust nonlinearity depending on whether it's a "top-k" or "random k" rule
            if self.nonlinearity_type == 'hard_k':
                self.y = hard_k_winner_take_all(self.logits, self.k)
            else: 
                self.y = random_k_winner_take_all(self.logits, self.k)
        # during retrieval, always pick the top k
        else:
            self.y = hard_k_winner_take_all(self.logits, self.k)


        out = self.W_yx @ self.y

        if phase == "learning":
            self.__adjust_weights(x)

        # sparsify retrieved output to have the same sparsity level as the original inputs
        return sparsify(out, self.input_sparsity)

    # perform a retrieval operation with the input 'x' with the weights frozen throughout
    def retrieve(self, x):
        return self.forward(x, phase="retrieval")

    '''
    Learn or retrieve from a sequence of patterns one-by-one
    Arguments:
    data        :       the sequence of binary patterns to learn, with shape num_data x input_size
    phase       :       whether to learn these patterns or simply perform retrieval using them as inputs
    '''
    def learn_patterns(self, data, phase="learning"):
        # store collection of obtained hidden and output patterns for each item in the sequence
        y_matrix = np.zeros((data.shape[0], self.hidden_size))
        out_matrix = np.zeros((data.shape[0], self.input_size))

        for i in range(data.shape[0]):
            x = data[i].reshape((-1, 1))
            out = self.forward(x, phase)
            y_matrix[i] = (self.y).reshape(-1)
            out_matrix[i] = out.reshape(-1)

        return y_matrix, out_matrix

    # shortcut to perform retrieval from a sequence of items one-by-one
    def retrieve_patterns(self, data):
        return self.learn_patterns(data, phase="retrieval")

    '''
    For a dataset of binary patterns, construct random partial patterns for each item
    and perform retrieval using these resulting partial patterns

    Arguments:
    data        :       the sequence of binary patterns to learn, with shape num_data x input_size
    cue_level   :       the proportion of total 1-bits in a given binary pattern that remain active
    '''
    def retrieve_from_partial_cues(self, data, cue_level):
        y_matrix = np.zeros((data.shape[0], self.hidden_size))
        out_matrix = np.zeros((data.shape[0], self.input_size))

        for i in range(data.shape[0]):
            age = data.shape[0] - i - 1
            # generate a partial pattern for the ith item in the dataset
            partial = get_partial_pattern(data, age, cue_level)
            partial = partial.reshape((-1, 1))

            # retrieve from the partial pattern
            out = self.retrieve(partial)
            out_matrix[i] = out.reshape(-1)
            y_matrix[i] = (self.y).reshape(-1)

        return y_matrix, out_matrix

    # Obtain a collection of hidden logits and sparsified hidden states across a sequence of items
    def retrieve_hidden_representations(self, data):
        logit_matrix = np.zeros((data.shape[0], self.hidden_size))
        y_matrix = np.zeros((data.shape[0], self.hidden_size))

        for i in range(data.shape[0]):
            x = data[i].reshape((-1, 1))
            out = self.forward(x, phase="retrieval")
            logit_matrix[i] = (self.logits).reshape(-1)
            y_matrix[i] = (self.y).reshape(-1)

        return logit_matrix, y_matrix

# the original MHN, as a reference -- this code is not used elsewhere!
class ModernHopfieldNet:
    def __init__(self, data, num_features, num_memories, slope_param, time_param, input_sparsity=0.1):
        self.train_data = data
        (rows, cols) = np.shape(data)
        assert num_features == cols
        assert num_memories == rows
        self.num_features = num_features
        self.num_memories = num_memories
        self.weights = data
        self.slope_param = slope_param
        self.time_param = time_param
        self.hidden_state = np.zeros(num_memories)
        self.input_sparsity = input_sparsity

    def forward(self, v, num_epochs):
        v_curr = v  # / np.linalg.norm(v)    # normalizing, though this condition isn't strictly necessary
        for t in range(num_epochs):
            h = np.matmul(self.weights, v_curr)
            weights_transpose = np.transpose(self.weights)
            softmax_h = softmax(h, self.slope_param)
            self.hidden_state = softmax_h
            v_new = v_curr + (1.0 / self.time_param) * (np.matmul(weights_transpose, softmax_h) - v_curr)
            v_curr = v_new

        return sparsify(v_curr, self.input_sparsity)

    def retrieve(self, v):
        return self.forward(v, num_epochs=1)

    def __record_distances_to_mems(self, v_curr):
        distances = np.zeros(self.weights.shape[0])
        for i in range(len(self.weights)):
            distances[i] = np.sqrt(np.sum((self.weights[i] - v_curr) ** 2))
        return distances

    def __find_stored_mem(self, v_curr, criterion):
        distances = self.__record_distances_to_mems(v_curr)
        indices = np.where(distances == np.min(distances))[0]
        if (distances[indices[0]] < criterion):
            return self.weights[indices[0]]
        else:
            return v_curr

    def retrieve_patterns(self, data):
        y_matrix = np.zeros((data.shape[0], self.hidden_size))
        out_matrix = np.zeros((data.shape[0], self.input_size))

        for i in range(data.shape[0]):
            x = data[i]
            out = self.forward(x, num_epochs=1)
            y_matrix[i] = (self.hidden_state).reshape(-1)
            out_matrix[i] = out.reshape(-1)

        return y_matrix, out_matrix

    def set_slope_param(self, slope_param):
        self.slope_param = slope_param

    def set_time_param(self, time_param):
        self.time_param = time_param

    def get_slope_param(self):
        return self.slope_param

    def get_time_param(self):
        return self.time_param
