import numpy as np
from numba import njit

from .hmm_functions import HMMParam, write_HMM_to_file, Simulate_transition
from .helper_functions import find_runs


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Make test data
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def set_seed(value):
    np.random.seed(value)


@njit(cache=True)
def simulatate_mutation_position(n_mutations):
    return np.random.choice(1000, n_mutations)


@njit(cache=True)
def simulate_poisson(lam):
    return np.random.poisson(lam)


def simulate_path(data_set_length, n_chromosomes, hmm_parameters, SEED):
    '''Create test data set of size data_set_length. Also create uniform weights and uniform mutation rates'''
    
    # Config
    np.random.seed(SEED)
    set_seed(SEED)
    
    total_size = data_set_length * n_chromosomes

    state_values = [x for x in range(len(hmm_parameters.state_names))]
    n_states = len(state_values)
    path = np.zeros(total_size, dtype = int)

    observations = np.zeros(total_size, dtype = int)
    weights = np.ones(total_size)
    mutrates = np.ones(total_size)  

    # Use prior dist if starting window
    current_state = np.random.choice(state_values, p=hmm_parameters.starting_probabilities)

    if n_states == 2:
        path, observations = simulate_two_state_path(current_state, total_size, hmm_parameters.transitions, hmm_parameters.emissions)
        return observations, mutrates, weights, path

    prevstate = current_state
    for index in range(total_size):
        
        if index > 0:
            current_state = Simulate_transition(n_states, hmm_parameters.transitions[prevstate,:], prevstate)

        observations[index] = simulate_poisson(hmm_parameters.emissions[current_state])
        path[index] = current_state
        prevstate = current_state
            
            
    return observations, mutrates, weights, path


@njit(cache=True)
def simulate_two_state_path(first_state, total_size, transitions, emissions):
    """
    Same random draws, in the same order, as the loop in simulate_path (Simulate_transition + simulate_poisson),
    but compiled.
    """
    path = np.zeros(total_size, dtype=np.int64)
    observations = np.zeros(total_size, dtype=np.int64)
    current_state = first_state
    for index in range(total_size):
        if index > 0:
            p_stay = min(transitions[current_state, current_state], 1.0)
            if np.random.binomial(1, p_stay) != 1:
                current_state = 1 - current_state
        observations[index] = np.random.poisson(emissions[current_state])
        path[index] = current_state
    return path, observations


def write_data(path, obs, data_set_length, n_chromosomes, hmm_parameters, SEED):

    # Config
    np.random.seed(SEED)
    set_seed(SEED)
    
    window_size = 1000
    bases = np.array(['A','C','G','T'])

    CHROMOSOMES = [f'chr{x + 1}' for x in range(n_chromosomes)] 
    CHROMOSOME_RUNS = []
    previous_start = 0
    for chrom in CHROMOSOMES:
        CHROMOSOME_RUNS.append([chrom, previous_start, data_set_length])
        previous_start += data_set_length



    # Make obs file and true simulated segments
    with open('obs.txt','w') as obs_file, open('simulated_segments.txt', 'w') as out:
        print('chrom', 'pos', 'ancestral_base', 'genotype', sep = '\t', file = obs_file)
        print('chrom', 'start', 'end', 'length', 'state', sep = '\t', file = out)

        for (chrom, chrom_start_index, chrom_length_index) in CHROMOSOME_RUNS:
            for (state_id, start_index, length_index) in find_runs(path[chrom_start_index:chrom_start_index + chrom_length_index]):

                state = hmm_parameters.state_names[state_id]
                genome_start = start_index * window_size
                genome_length =  length_index * window_size
                genome_end = genome_start + genome_length
                print(chrom, genome_start, genome_end, genome_length, state, sep = '\t', file = out)

                # write mutations
                n_mutations_segment = obs[(chrom_start_index + start_index):(chrom_start_index + start_index + length_index)]
                for index, n_mutations in enumerate(n_mutations_segment):

                    if n_mutations == 0:
                        continue

                    for random_int in simulatate_mutation_position(n_mutations):
                        mutation = (start_index + index) * window_size + random_int
                        ancestral_base, derived_base = np.random.choice(bases, 2, replace = False)
                        
                        print(chrom, mutation, ancestral_base, ancestral_base + derived_base, sep = '\t', file = obs_file)

    # Make weights file and mutation file
    with open('weights.bed','w') as weights_file, open('mutrates.bed','w') as mutrates_file:
        for chrom in CHROMOSOMES:
            print(chrom, 0, data_set_length * window_size, sep = '\t', file = weights_file)
            print(chrom, 0, data_set_length * window_size, 1, sep = '\t', file = mutrates_file)

    # Make initial guesses
    initial_guess = HMMParam(['Human', 'Archaic'], [0.5, 0.5], [[0.99,0.01],[0.02,0.98]], [0.03, 0.3]) 
    write_HMM_to_file(initial_guess, 'Initialguesses.json')

                  
    return 

