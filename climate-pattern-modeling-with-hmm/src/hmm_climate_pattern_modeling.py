# Climate Pattern Modeling with Hidden Markov Model
# Water Programming: A Collaborative Research Blog (Fitting HMMs: Background and Methods)
# https://waterprogramming.wordpress.com/2018/07/03/fitting-hidden-markov-models-part-i-background-and-methods/

# Import Statements
import numpy as np
from scipy.stats import norm
import os

# Functions Definition

# Generating Stationary Distribution from Transition Distribution
# Markov Chains: Stationary Distributions
# https://www.stat.berkeley.edu/~mgoldman/Section0220.pdf
def stationary_distribution(transition_distribution):
    num_states = transition_distribution.shape[0]
    
    delta = np.empty((num_states, num_states))
    delta_i = np.empty((num_states, num_states))
    stationary_distribution = np.empty(num_states)
    
    for i in range(num_states - 1):
        delta[i] = transition_distribution[:, i]
        delta[i, i] = delta[i, i] - 1
    
    delta[num_states - 1] = np.ones(num_states)
    
    for i in range(num_states):
        delta_i[:, :] = delta
        delta_i[:, i] = np.concatenate((np.zeros(num_states - 1), np.ones(1)))
        
        # ref: https://www.geeksforgeeks.org/how-to-calculate-the-determinant-of-a-matrix-using-numpy/
        stationary_distribution[i] = np.linalg.det(delta_i) / np.linalg.det(delta)
        
    return stationary_distribution

# Viterbi Algorithm Implementation
def viterbi(observations, transition_distribution, emission_means, emission_std_devs):
    num_observations = observations.shape[0]
    num_states = transition_distribution.shape[0]
    
    initial_distribution = stationary_distribution(transition_distribution)
    
    likelihood_previous = np.empty(num_states)
    likelihood_current = np.empty(num_states)
    
    previous_node = np.full((num_states, num_observations), -1)
    
    for i in range(num_states):
        # ref: https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.stats.norm.html
        # ref: https://www.kite.com/python/docs/scipy.stats.norm.pdf
        likelihood_previous[i] = np.log(initial_distribution[i]) + np.log(norm.pdf(x=observations[0], loc=emission_means[i], scale=emission_std_devs[i]))
    
    for i in range(1, num_observations):
        for j in range(num_states):
            likelihood_current[j] = likelihood_previous[0] + np.log(transition_distribution[0, j]) + np.log(norm.pdf(x=observations[i], loc=emission_means[j], scale=emission_std_devs[j]))
            previous_node[j, i] = 0
            
            for k in range(1, num_states):
                likelihood = likelihood_previous[k] + np.log(transition_distribution[k, j]) + np.log(norm.pdf(x=observations[i], loc=emission_means[j], scale=emission_std_devs[j]))
                
                if likelihood > likelihood_current[j]:
                    likelihood_current[j] = likelihood
                    previous_node[j, i] = k
        
        likelihood_previous[:] = likelihood_current
    
    estimated_states_sequence = [likelihood_current.argmax()]
    
    for i in range(num_observations - 1):
        estimated_states_sequence.append(previous_node[estimated_states_sequence[i], num_observations - (i + 1)])
        
    estimated_states_sequence.reverse()
    return estimated_states_sequence

# Baum-Welch Learning Algorithm Implementation
def forward(observations, transition_distribution, emission_means, emission_std_devs):
    num_observations = observations.shape[0]
    num_states = transition_distribution.shape[0]
    
    initial_distribution = stationary_distribution(transition_distribution)
    
    forward_matrix = np.zeros((num_states, num_observations))
    
    for i in range(num_states):
        forward_matrix[i, 0] = initial_distribution[i] * norm.pdf(x=observations[0], loc=emission_means[i], scale=emission_std_devs[i])
    
    forward_matrix[:, 0] = forward_matrix[:, 0] / np.sum(forward_matrix[:, 0])
    
    for i in range(1, num_observations):
        for j in range(num_states):
            for k in range(num_states):
                forward_matrix[j, i] = forward_matrix[j, i] + forward_matrix[k, i - 1] * transition_distribution[k, j] * norm.pdf(x=observations[i], loc=emission_means[j], scale=emission_std_devs[j])
        
        forward_matrix[:, i] = forward_matrix[:, i] / np.sum(forward_matrix[:, i])
    
    return forward_matrix

def backward(observations, transition_distribution, emission_means, emission_std_devs):
    num_observations = observations.shape[0]
    num_states = transition_distribution.shape[0]
    
    backward_matrix = np.zeros((num_states, num_observations))
    backward_matrix[:, num_observations - 1] = np.ones(num_states)
    
    for i in range(1, num_observations):
        for j in range(num_states):
            for k in range(num_states):
                backward_matrix[j, num_observations - (i + 1)] = backward_matrix[j, num_observations - (i + 1)] + backward_matrix[k, num_observations - i] * transition_distribution[j, k] * norm.pdf(x=observations[num_observations - i], loc=emission_means[k], scale=emission_std_devs[k])
        
        backward_matrix[:, num_observations - (i + 1)] = backward_matrix[:, num_observations - (i + 1)] / np.sum(backward_matrix[:, num_observations - (i + 1)])
    
    return backward_matrix

def baum_welch_learning(observations, transition_distribution, emission_means, emission_std_devs, num_epochs):
    num_observations = observations.shape[0]
    num_states = transition_distribution.shape[0]
    
    responsibility_profile_for_states = np.empty((num_states, num_observations))
    responsibility_profile_for_transitions = np.empty((num_states, num_states, num_observations - 1))
    
    for epoch in range(num_epochs):
        forward_matrix = forward(observations, transition_distribution, emission_means, emission_std_devs)
        backward_matrix = backward(observations, transition_distribution, emission_means, emission_std_devs)
        
        for i in range(num_observations):
            for j in range(num_states):
                responsibility_profile_for_states[j, i] = forward_matrix[j, i] * backward_matrix[j, i]
        
        responsibility_profile_for_states = responsibility_profile_for_states / np.sum(responsibility_profile_for_states, axis=0)
        
        for i in range(num_observations - 1):
            for j in range(num_states):
                for k in range(num_states):
                    responsibility_profile_for_transitions[j, k, i] = forward_matrix[j, i] * transition_distribution[j, k] * norm.pdf(x=observations[i + 1], loc=emission_means[k], scale=emission_std_devs[k]) * backward_matrix[k, i + 1]
        
        responsibility_profile_for_transitions = responsibility_profile_for_transitions / np.sum(responsibility_profile_for_transitions, axis=(0, 1))
        
        transition_distribution = np.sum(responsibility_profile_for_transitions, axis=-1)
        transition_distribution = transition_distribution / np.sum(transition_distribution, axis=1)[:, None]
        
        emission_means = np.sum(responsibility_profile_for_states * observations, axis=1) / np.sum(responsibility_profile_for_states, axis=1)
        emission_std_devs = np.sqrt(np.sum(responsibility_profile_for_states * np.power(np.tile(observations, (num_states, 1)) - emission_means.reshape(num_states, 1), 2), axis=1) / np.sum(responsibility_profile_for_states, axis=1))
    
    return transition_distribution, emission_means, emission_std_devs

# Experimentations

# Extracting Rainfall Estimates Data from `data.txt`
with open('./inputdir/data.txt', 'r') as observations_file:
    observations = observations_file.read().split('\n')

observations = np.array([float(observation) for observation in observations if observation != ''])

# Extracting HMM Parameters from `parameters.txt`
with open('./inputdir/parameters.txt', 'r') as parameters_file:
    parameters = parameters_file.read().split('\n')

parameters = [parameter_line for parameter_line in parameters if parameter_line != '']

num_states = int(parameters[0])

transition_distribution = np.empty((num_states, num_states))

for i in range(num_states):
    transition_distribution[i] = np.array(parameters[i + 1].split('\t'), dtype=np.float64)
    
gaussian_distribution_means = np.array(parameters[num_states + 1].split('\t'), dtype=np.float64)
gaussian_distribution_std_devs = np.sqrt(np.array(parameters[num_states + 2].split('\t'), dtype=np.float64))

# Estimating States Sequence with Provided Parameters by Solving Decoding Problem
if not os.path.exists('./outputdir'):
    os.makedirs('./outputdir')

estimated_states_sequence = viterbi(observations, transition_distribution, gaussian_distribution_means, gaussian_distribution_std_devs)
estimated_states_sequence = ['"El Nino"' if estimated_state == 0 else '"La Nina"' for estimated_state in estimated_states_sequence]

with open('./outputdir/estimated-states-sequence-with-provided-parameters.txt', 'w') as output_file:
    output_file.write('\n'.join(estimated_states_sequence))

# Estimating HMM Parameters with Baum-Welch Learning Algorithm
transition_distribution, gaussian_distribution_means, gaussian_distribution_std_devs = baum_welch_learning(observations, transition_distribution, gaussian_distribution_means, gaussian_distribution_std_devs, num_epochs=5)

num_states = transition_distribution.shape[0]
learned_parameters = [str(num_states)]

for i in range(num_states):
    learned_parameters.append('\t'.join(list(map(str, transition_distribution[i].tolist()))))

learned_parameters.append('\t'.join(list(map(str, gaussian_distribution_means.tolist()))))
learned_parameters.append('\t'.join(list(map(str, np.power(gaussian_distribution_std_devs, 2).tolist()))))
learned_parameters.append('\t'.join(list(map(str, stationary_distribution(transition_distribution).tolist()))))

with open('./outputdir/learned-parameters.txt', 'w') as output_file:
    output_file.write('\n'.join(learned_parameters))

# Estimating States Sequence with Learned Parameters by Solving Decoding Problem
estimated_states_sequence = viterbi(observations, transition_distribution, gaussian_distribution_means, gaussian_distribution_std_devs)
estimated_states_sequence = ['"El Nino"' if estimated_state == 0 else '"La Nina"' for estimated_state in estimated_states_sequence]

with open('./outputdir/estimated-states-sequence-with-learned-parameters.txt', 'w') as output_file:
    output_file.write('\n'.join(estimated_states_sequence))
