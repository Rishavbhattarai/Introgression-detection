import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import sys

from hmm_functions import Hybrid_path, Emission_probs_poisson, Calculate_Posterior_probabillities, calculate_log
from make_test_data import simulate_path, set_seed

@njit
def Speed_up_likelihood(trans, emission):
    return calculate_log(trans * emission)

@njit
def Calculate_loglikelihood(path, emissions, starting_probabilities, transitions):

    loglikelihood = 0
    for i, state in enumerate(path):
        if i == 0:
            loglikelihood += Speed_up_likelihood(starting_probabilities[state], emissions[i,state])
        else:
            loglikelihood += Speed_up_likelihood(transitions[path[i-1], state], emissions[i,state])

    return loglikelihood

def scale_values(values):

    min_value = min(values)
    max_value = max(values)

    if max_value - min_value == 0:
        sys.exit('All your values are exactly the same - try increasing the window size!')

    scaled_values = [(x - min_value) / (max_value - min_value) for x in values]
    return scaled_values


def distance_to_line(x, y, intercept, slope):
    return abs((slope * x) - y + intercept) / np.sqrt(slope ** 2 + 1)


def compute_confusion_matrix(true_array, pred_array, states):

    n_states = len(states)
    confusion_matrix = np.zeros((n_states, n_states))

    for true, pred in zip(true_array, pred_array):
        confusion_matrix[true, pred] += 1

    confusion_matrix_as_list = np.asarray(confusion_matrix).reshape(-1)

    return confusion_matrix_as_list



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def Find_best_alpha(obs, mutrates, weights, hmm_parameters, START, END, STEPS):

    emissions = Emission_probs_poisson(hmm_parameters.emissions, obs, weights, mutrates)
    posterior_probs = Calculate_Posterior_probabillities(emissions, hmm_parameters)
    logged_posterior_probs = np.log(posterior_probs.T)
    
    x_coordinates = []
    y_coordinates = []
    alphas = []

    for ALPHA in np.linspace(START, END, STEPS):
        
        decoded_path = Hybrid_path(emissions, hmm_parameters.starting_probabilities, hmm_parameters.transitions, logged_posterior_probs, ALPHA)
        average_path_posterior_probability = np.mean(posterior_probs[decoded_path, np.arange(posterior_probs.shape[1])])
        loglikelihood = Calculate_loglikelihood(decoded_path, emissions, hmm_parameters.starting_probabilities, hmm_parameters.transitions)

        x_coordinates.append(average_path_posterior_probability)
        y_coordinates.append(loglikelihood)          
        alphas.append(ALPHA)

    return x_coordinates, y_coordinates, alphas





def Find_best_alpha_simulated(hmm_parameters, data_set_length, START, END, STEPS, SEED):

    # Config
    np.random.seed(SEED)
    set_seed(SEED)

    # simulate data
    obs, mutrates, weights, path = simulate_path(data_set_length, 1, hmm_parameters, SEED)    
    
    emissions = Emission_probs_poisson(hmm_parameters.emissions, obs, weights, mutrates)
    posterior_probs = Calculate_Posterior_probabillities(emissions, hmm_parameters)
    logged_posterior_probs = np.log(posterior_probs.T)

    x_coordinates = []
    y_coordinates = []
    alphas = []
    confusion_lists = []

    for ALPHA in np.linspace(START, END, STEPS):
        decoded_path = Hybrid_path(emissions, hmm_parameters.starting_probabilities, hmm_parameters.transitions, logged_posterior_probs, ALPHA)
        average_path_posterior_probability = np.mean(posterior_probs[decoded_path, np.arange(posterior_probs.shape[1])])
        loglikelihood = Calculate_loglikelihood(decoded_path, emissions, hmm_parameters.starting_probabilities, hmm_parameters.transitions)

        x_coordinates.append(average_path_posterior_probability)
        y_coordinates.append(loglikelihood)          
        alphas.append(ALPHA)
        
        confusion_lists.append(compute_confusion_matrix(path, decoded_path, hmm_parameters.state_names))

    return x_coordinates, y_coordinates, alphas, confusion_lists




def Make_Artemis_Plot(x_coordinates, y_coordinates, alphas, out_plot_file):

    # Find best alpha values
    scaled_x_coordinates = scale_values(x_coordinates)
    scaled_y_coordinates = scale_values(y_coordinates)

    best_dist, best_alpha = np.inf, 0
    for (alpha, x, y) in zip(alphas, scaled_x_coordinates, scaled_y_coordinates):
        dist = distance_to_line(x, y, 0, 1)
        if dist < best_dist:
            best_dist, best_alpha = dist, alpha
    

    plt.plot(scaled_x_coordinates, scaled_y_coordinates, color="grey") 
    plt.plot(np.linspace(0, 1, 101), np.linspace(0, 1, 101), '--', color="black")

    plt.xlabel('Average pointwise posterior probability')
    plt.ylabel('Log joint posterior probability')
    plt.title(f'Best alpha = {round(best_alpha,4)}')
    
    plt.grid(True)  
    plt.savefig(out_plot_file, bbox_inches='tight', dpi = 1000)





def Write_Artemis_output(hmm_parameters, x_coordinates, y_coordinates, alphas, output_file, confusion_list_alphas):
     
    HEADER = ['alpha', 'average_pointwise_posterior_probability','loglikelihood']
    if confusion_list_alphas is not None:
        n_states = len(hmm_parameters.state_names)
        state_names = hmm_parameters.state_names
        state_transition_names = [f'{state_names[x]}_{state_names[y]}' for x in range(n_states) for y in range(n_states)]
        HEADER += state_transition_names

    with open(output_file, 'w') as out:
            
        print('\t'.join(HEADER), file = out)
        for index, (alpha, x, y) in enumerate(zip(alphas, x_coordinates, y_coordinates)):

            if confusion_list_alphas is not None:
                confusion_list = confusion_list_alphas[index]
            else:
                confusion_list = []
            print(round(alpha, 3), round(x,6), y, '\t'.join([str(x) for x in confusion_list]), sep = '\t', file = out)

            
