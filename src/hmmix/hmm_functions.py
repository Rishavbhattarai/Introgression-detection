from collections import defaultdict
import numpy as np
from numba import njit
import json
from .helper_functions import find_runs, Annotate_with_ref_genome, Make_folder_if_not_exists, flatten_list

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# HMM Parameter Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class HMMParam:
    def __init__(self, state_names, starting_probabilities, transitions, emissions): 
        self.state_names = np.array(state_names)
        self.starting_probabilities = np.array(starting_probabilities)
        self.transitions = np.array(transitions)
        self.emissions = np.array(emissions)

    def __str__(self):
        out = f'> state_names = {self.state_names.tolist()}\n'
        out += f'> starting_probabilities = {np.matrix.round(self.starting_probabilities, 3).tolist()}\n'
        out += f'> transitions = {np.matrix.round(self.transitions, 3).tolist()}\n'
        out += f'> emissions = {np.matrix.round(self.emissions, 3).tolist()}'
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.state_names}, {self.starting_probabilities}, {self.transitions}, {self.emissions})'
        
# Read HMM parameters from a json file
def read_HMM_parameters_from_file(filename):

    if filename is None:
        return get_default_HMM_parameters()

    with open(filename) as json_file:
        data = json.load(json_file)

    return HMMParam(state_names = data['state_names'], 
                    starting_probabilities = data['starting_probabilities'], 
                    transitions = data['transitions'], 
                    emissions = data['emissions'])

# Set default parameters
def get_default_HMM_parameters():
    return HMMParam(state_names = ['Human', 'Archaic'], 
                    starting_probabilities = [0.98, 0.02], 
                    transitions = [[0.9999,0.0001],[0.02,0.98]], 
                    emissions = [0.04, 0.4])

# Save HMMParam to a json file
def write_HMM_to_file(hmmparam, outfile):
    data = {key: value.tolist() for key, value in vars(hmmparam).items()}
    json_string = json.dumps(data, indent = 2) 
    with open(outfile, 'w') as out:
        out.write(json_string)




# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# HMM functions
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def poisson_probability_underflow_safe(n, lam):
    # naive:   np.exp(-lam) * lam**n / factorial(n)

    # iterative, to keep the components from getting too large or small:
    p = np.exp(-lam)
    for i in range(n):
        p *= lam
        p /= i+1
    return p

@njit(cache=True)
def Emission_probs_poisson(emissions, observations, weights, mutrates):
    n = len(observations)
    n_states = len(emissions)
    
    probabilities = np.zeros( (n, n_states) ) 
    for state in range(n_states): 
        for index in range(n):
            lam = emissions[state] * weights[index] * mutrates[index]
            probabilities[index,state] = poisson_probability_underflow_safe(observations[index], lam)

    return probabilities

@njit(cache=True)
def fwd_step(alpha_prev, E, trans_mat):
    n_states = len(alpha_prev)
    alpha_new = np.zeros(n_states)
    for j in range(n_states):
        acc = 0.0
        for i in range(n_states):
            acc += alpha_prev[i] * trans_mat[i, j]
        alpha_new[j] = acc * E[j]
    n = np.sum(alpha_new)
    return alpha_new / n, n

@njit(cache=True)
def forward(probabilities, transitions, init_start):
    n, n_states = probabilities.shape
    forwards_in = np.zeros( (n, n_states) ) 
    scale_param = np.ones(n)

    for t in range(n):
        if t == 0:
            forwards_in[t,:] = init_start  * probabilities[t,:]
            scale_param[t] = np.sum( forwards_in[t,:])
            forwards_in[t,:] = forwards_in[t,:] / scale_param[t]  
        else:
            # same arithmetic as fwd_step, written in place
            total = 0.0
            for j in range(n_states):
                acc = 0.0
                for i in range(n_states):
                    acc += forwards_in[t-1, i] * transitions[i, j]
                forwards_in[t, j] = acc * probabilities[t, j]
                total += forwards_in[t, j]
            scale_param[t] = total
            for j in range(n_states):
                forwards_in[t, j] = forwards_in[t, j] / total

    return forwards_in, scale_param

@njit(cache=True)
def bwd_step(beta_next, E, trans_mat, n):
    n_states = len(beta_next)
    beta = np.zeros(n_states)
    for i in range(n_states):
        acc = 0.0
        for j in range(n_states):
            acc += (trans_mat[i, j] * E[j]) * beta_next[j]
        beta[i] = acc / n
    return beta

@njit(cache=True)
def backward(emission_probs, transitions, scales):
    n, n_states = emission_probs.shape
    beta = np.ones((n, n_states))
    for t in range(n - 1, 0, -1):
        # same as bwd_step(beta[t,:], emission_probs[t,:], transitions, scales[t]), in place
        for i in range(n_states):
            acc = 0.0
            for j in range(n_states):
                acc += (transitions[i, j] * emission_probs[t, j]) * beta[t, j]
            beta[t - 1, i] = acc / scales[t]
    return beta


def forward_pass(hmm_parameters, weights, obs, mutrates):
    """Emission probabilities, scaled forward probabilities and scales for the given parameters"""
    emission_probs = Emission_probs_poisson(hmm_parameters.emissions, obs, weights, mutrates)
    forward_probs, scales = forward(emission_probs, hmm_parameters.transitions, hmm_parameters.starting_probabilities)
    return emission_probs, forward_probs, scales


def GetProbability(hmm_parameters, weights, obs, mutrates):

    _, _, scales = forward_pass(hmm_parameters, weights, obs, mutrates)
    forward_probility_of_obs = np.sum(np.log(scales))

    return forward_probility_of_obs


@njit(cache=True)
def fwd_step_keep_track(alpha_prev, E, trans_mat, results, back_track_states):
    """One Viterbi step, written into results and back_track_states (no allocations)"""
    
    # scaling factor: sum((alpha_prev @ trans_mat) * E)
    n = 0.0
    for j in range(len(E)):
        acc = 0.0
        for i in range(len(E)):
            acc += alpha_prev[i] * trans_mat[i, j]
        n += acc * E[j]
    
    for current_s in range(len(E)):
        results[current_s] = 0.0
        back_track_states[current_s] = 0
        for prev_s in range(len(E)):
            new_prob = alpha_prev[prev_s] * trans_mat[prev_s, current_s] * E[current_s] / n

            if new_prob > results[current_s]:
                results[current_s] = new_prob
                back_track_states[current_s] = prev_s


@njit(cache=True)
def viterbi(probabilities, transitions, init_start):
    n = len(probabilities)
    forwards_in = np.zeros( (n, len(init_start)) ) 
    backtracks = np.zeros( (n, len(init_start)), dtype=np.int32) 

    for t in range(n):
        if t == 0:
            forwards_in[t,:] = init_start  * probabilities[t,:]
            scale_param = np.sum( forwards_in[t,:])
            forwards_in[t,:] = forwards_in[t,:] / scale_param
        else:
            fwd_step_keep_track(forwards_in[t-1,:], probabilities[t,:], transitions, forwards_in[t,:], backtracks[t,:])

    return forwards_in, backtracks


@njit(cache=True)
def calculate_log(x):
	return np.log(x)


@njit(cache=True)
def hybrid_step(prev, alpha, em, trans):
    value = prev + alpha * calculate_log(em * trans)
    best_state = np.argmax(value)
    max_prob = value[best_state]

    return best_state, max_prob


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def logoutput(hmm_parameters, loglikelihood, iteration):

    n_states = len(hmm_parameters.emissions)

    # Make header
    if iteration == 0:    
        print_emissions = '\t'.join(['emis{0}'.format(x + 1) for x in range(n_states)])
        print_starting_probabilities = '\t'.join(['start{0}'.format(x + 1) for x in range(n_states)])
        print_transitions = '\t'.join(['trans{0}_{0}'.format(x + 1) for x in range(n_states)])
        print('iteration', 'loglikelihood', print_starting_probabilities, print_emissions, print_transitions, sep = '\t')

    # Print parameters
    print_emissions = '\t'.join([str(x) for x in np.matrix.round(hmm_parameters.emissions, 4)])
    print_starting_probabilities = '\t'.join([str(x) for x in np.matrix.round(hmm_parameters.starting_probabilities, 3)])
    print_transitions = '\t'.join([str(x) for x in np.matrix.round(hmm_parameters.transitions, 4).diagonal()])
    print(iteration, round(loglikelihood, 4), print_starting_probabilities, print_emissions, print_transitions, sep = '\t')



def TrainBaumWelsch(hmm_parameters, weights, obs, mutrates, forward_results = None):
    """
    Trains the model once, using the forward-backward algorithm. 

    forward_results can be the output of forward_pass() for hmm_parameters, to avoid computing it twice.
    """

    n_states = len(hmm_parameters.starting_probabilities)

    if forward_results is None:
        forward_results = forward_pass(hmm_parameters, weights, obs, mutrates)
    emission_probs, forward_probs, scales = forward_results
    backward_probs = backward(emission_probs, hmm_parameters.transitions, scales)

    # Update starting probs
    posterior_probs = forward_probs * backward_probs
    normalize = np.sum(posterior_probs)
    new_starting_probabilities = np.sum(posterior_probs, axis=0)/normalize 

    # Update emission
    new_emissions_matrix = np.zeros((n_states))
    for state in range(n_states):
        top = np.sum(posterior_probs[:,state] * obs)
        bottom = np.sum(posterior_probs[:,state] * (weights * mutrates) )
        new_emissions_matrix[state] = top/bottom

        # do variance
        #variance = np.sum(posterior_probs[:,state] * (new_emissions_matrix[state] - obs )**2 ) / bottom
        #print(state, new_emissions_matrix[state], variance)

    # Update Transition probs 
    new_transitions_matrix =  np.zeros((n_states, n_states))
    for state1 in range(n_states):
        for state2 in range(n_states):
            new_transitions_matrix[state1,state2] = np.sum( forward_probs[:-1,state1]  * backward_probs[1:,state2]  * hmm_parameters.transitions[state1, state2] * emission_probs[1:,state2]/ scales[1:] )

    new_transitions_matrix /= new_transitions_matrix.sum(axis=1)[:,np.newaxis]

    return HMMParam(hmm_parameters.state_names,new_starting_probabilities, new_transitions_matrix, new_emissions_matrix)


def TrainModel(obs, mutrates, weights, hmm_parameters, epsilon, maxiterations):


    # Get probability of data with initial parameters
    forward_results = forward_pass(hmm_parameters, weights, obs, mutrates)
    previous_loglikelihood = np.sum(np.log(forward_results[2]))
    logoutput(hmm_parameters, previous_loglikelihood, 0)
    
    # Train parameters using Baum Welch algorithm
    # (the forward pass used for the likelihood is reused by the next Baum-Welch step)
    for i in range(1,maxiterations):
        hmm_parameters = TrainBaumWelsch(hmm_parameters, weights, obs, mutrates, forward_results)
        forward_results = forward_pass(hmm_parameters, weights, obs, mutrates)
        new_loglikelihood = np.sum(np.log(forward_results[2]))
        logoutput(hmm_parameters, new_loglikelihood, i)

        if new_loglikelihood - previous_loglikelihood < epsilon:       
            break 

        previous_loglikelihood = new_loglikelihood

    # Write the optimal parameters
    return hmm_parameters





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Decode (different algorithms)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def Calculate_Posterior_probabillities(emission_probs, hmm_parameters):
    """Get posterior probability of being in state s at time t"""
    
    forward_probs, scales = forward(emission_probs, hmm_parameters.transitions, hmm_parameters.starting_probabilities)
    backward_probs = backward(emission_probs, hmm_parameters.transitions, scales)
    posterior_probabilities = (forward_probs * backward_probs).T

    return posterior_probabilities


def PMAP_path(posterior_probabilities):
    """Get maximum posterior decoding path"""
    path = np.argmax(posterior_probabilities, axis = 0)
    return path 


def Viterbi_path(emission_probs, hmm_parameters):
    """Get Viterbi path - aka most likeli path"""
    n_obs, _ = emission_probs.shape
    
    viterbi_probs, backtracks = viterbi(emission_probs, hmm_parameters.transitions, hmm_parameters.starting_probabilities)
    
    # backtracking
    viterbi_path = np.zeros(n_obs, dtype = int)
    viterbi_path[-1] = np.argmax(viterbi_probs[-1,:])
    for t in range(n_obs - 2, -1, -1):
        viterbi_path[t] = backtracks[t + 1, viterbi_path[t + 1]]

    return viterbi_path

@njit(cache=True)
def Hybrid_path(emission_probs, starting_probs, trans_matrix, logged_posterior_probabilities, ALPHA):
    """
    Decodes the model using the hybrid method. When alpha parameter is 0 this is the same as posterior and when it is 1 it is the same as viterbi
    """
    n_obs, n_states = emission_probs.shape
    BETA = 1 - ALPHA

    # for backtracking the hybrid method
    delta = np.zeros((n_obs, n_states))
    psi = np.zeros((n_obs - 1, n_states), dtype = np.int32)

    # initialize
    delta[0,:] = ALPHA * calculate_log(starting_probs * emission_probs[0,:]) + BETA * logged_posterior_probabilities[0,:]

    for t in range(1, n_obs):
        for state in range(n_states):
            # same as hybrid_step(delta[t-1, :], ALPHA, emission_probs[t,state], trans_matrix[:, state]), without temporary arrays
            best_state = 0
            max_prob = delta[t-1, 0] + ALPHA * np.log(emission_probs[t,state] * trans_matrix[0, state])
            for prev_state in range(1, n_states):
                value = delta[t-1, prev_state] + ALPHA * np.log(emission_probs[t,state] * trans_matrix[prev_state, state])
                if value > max_prob:
                    best_state, max_prob = prev_state, value
            delta[t,state] = max_prob + BETA * logged_posterior_probabilities[t,state]
            psi[t-1,state] = best_state
        

    # backtracking
    path = np.zeros(n_obs, dtype = np.int32)
    path[-1] = np.argmax(delta[-1,:])    
    for i in range(n_obs - 2, -1, -1):
        path[i] = psi[i, path[i + 1]]
    
    return path



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# inhomogeneous markov chain
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def Simulate_values(p):
    return np.random.binomial(1, p)

def Simulate_transition(n_states, matrix, current_state):

    # Numpy has some decimal issues with small numbers
    if matrix[current_state] > 1:
        matrix[current_state] = 1

    # prob of staying (can be done with numba so its quick)
    next_state = Simulate_values(matrix[current_state])
    if next_state == 1:
        return current_state
    else:
        if n_states == 2:
            return abs(current_state - 1) # if 1 - 1 = 0 or if 0 - 1 = 1
        else:
            new_matrix = [matrix[x] for x in range(n_states) if current_state != x]
            new_matrix /= np.sum(new_matrix)
            new_states = [x for x in range(n_states) if current_state != x]

            return np.random.choice(new_states, p=new_matrix)



def Make_inhomogeneous_transition_matrix(emission_probs, hmm_parameters):
    """
    Calculate transition matrix for each position in the sequence (given the data)
    """ 

    n_obs, n_states = emission_probs.shape
    _, scales = forward(emission_probs, hmm_parameters.transitions, hmm_parameters.starting_probabilities)
    backward_probs = backward(emission_probs, hmm_parameters.transitions, scales)

    # Make and initialise new transition matrix
    new_transition_matrix = np.zeros((n_obs, n_states, n_states))

    # starting probabilities
    sim_starting_probabilities = np.zeros(n_states)
    for state in range(n_states):
        sim_starting_probabilities[state] = hmm_parameters.starting_probabilities[state] * backward_probs[0, state] * emission_probs[0, state] / scales[0]
        new_transition_matrix[0, state, state] = sim_starting_probabilities[state]



    for state in range(n_states):
        for otherstate in range(n_states):
            new_transition_matrix[1:, otherstate, state] = (backward_probs[1:, state] ) / (backward_probs[:-1, otherstate] * scales[1:]) * hmm_parameters.transitions[otherstate, state] * emission_probs[1:, state]
    
    # normalize
    new_transition_matrix[1:] /= new_transition_matrix[1:].sum(axis=2, keepdims=True)
    

    return sim_starting_probabilities, new_transition_matrix



@njit(cache=True)
def simulate_two_state_chain(first_state, stay_probabilities):
    """
    Two-state Markov chain where stay_probabilities[t, s] is the probability of staying in state s at step t.
    Draws the same random numbers in the same order as calling Simulate_transition in a loop.
    """
    n = stay_probabilities.shape[0]
    path = np.zeros(n, dtype=np.int64)
    current_state = first_state
    path[0] = current_state
    for t in range(1, n):
        p_stay = min(stay_probabilities[t, current_state], 1.0)
        if np.random.binomial(1, p_stay) != 1:
            current_state = 1 - current_state
        path[t] = current_state
    return path


def Simulate_from_transition_matrix(sim_starting_probabilities, new_transition_matrix):


    number_observations, n_states, _, = new_transition_matrix.shape
    sim_path = np.zeros(number_observations, dtype=int) 
        
    # set start state
    current_state = np.random.choice(n_states, p=sim_starting_probabilities)
    sim_path[0] = current_state

    if n_states == 2:
        stay_probabilities = np.stack([new_transition_matrix[:, 0, 0], new_transition_matrix[:, 1, 1]], axis=1)
        return simulate_two_state_chain(current_state, stay_probabilities)

    for t in range(1, number_observations):
        next_state = Simulate_transition(n_states, new_transition_matrix[t,current_state,:], current_state)
        sim_path[t] = next_state
        current_state = next_state


    return sim_path



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Write segments to output file
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Rows are formatted in chunks: np.round is what round(np.float64, 4) uses, and str() of
# Python floats is the same text print() gives for numpy float64 values.
WRITE_CHUNK_SIZE = 100_000


def _format_columns(values):
    """Round a (n, k) array to 4 decimals and return one tab-separated string per row"""
    return ['\t'.join(map(str, row)) for row in np.round(values, 4).tolist()]


def Write_inhomogeneous_transition_matrix(chroms, starts, weights, mutrates, variants, hmm_parameters, new_transition_matrix, filename):
    n_obs = len(chroms)

    with open(filename, 'w') as out:
        state_combs = []
        for state1 in hmm_parameters.state_names:
            for state2 in hmm_parameters.state_names:
                state_combs.append(f'{state1}_{state2}')
        
        state_combs = '\t'.join(state_combs)
        print('chrom', 'start', 'called_sequence', 'mutationrate', state_combs,'variants', sep = '\t', file = out)

        for lo in range(0, n_obs, WRITE_CHUNK_SIZE):
            hi = min(lo + WRITE_CHUNK_SIZE, n_obs)
            # full precision (not rounded)
            transvalues = ['\t'.join(map(str, row)) for row in new_transition_matrix[lo:hi].reshape(hi - lo, -1).tolist()]
            rows = zip(chroms[lo:hi], np.asarray(starts[lo:hi]).tolist(), np.asarray(weights[lo:hi]).tolist(), np.asarray(mutrates[lo:hi]).tolist(), transvalues, variants[lo:hi])
            out.write(''.join(f'{chrom}\t{start}\t{w}\t{m}\t{trans}\t{var}\n' for chrom, start, w, m, trans, var in rows))


def Write_posterior_probs(chroms, starts, weights, mutrates, post_seq, path, variants, hmm_parameters, outfile, admixpop_file, obs_file):
    post_seq = post_seq.T
    n_obs = len(chroms)
    state_names = [str(x) for x in hmm_parameters.state_names]

    annotated_header = ''

    # Load archaic data
    if admixpop_file is not None:
        admix_pop_variants, _ = Annotate_with_ref_genome(admixpop_file, obs_file)
        annotated_header = '\tshared_with'

    def archaic_variants_by_position(chrom, var):
        if admixpop_file is None or var == '':
            return ''
        return ','.join(admix_pop_variants[f'{chrom}_{snp}'] for snp in var.split(','))

    with open(outfile, 'w') as out:
        print('chrom', 'start', 'called_sequence', 'mutationrate', '\t'.join(state_names),'state',f'variants{annotated_header}', sep = '\t', file = out)

        for lo in range(0, n_obs, WRITE_CHUNK_SIZE):
            hi = min(lo + WRITE_CHUNK_SIZE, n_obs)
            posteriors = _format_columns(post_seq[lo:hi])
            rows = zip(chroms[lo:hi], np.asarray(starts[lo:hi]).tolist(), np.asarray(weights[lo:hi]).tolist(), np.asarray(mutrates[lo:hi]).tolist(), posteriors, np.asarray(path[lo:hi]).tolist(), variants[lo:hi])
            out.write(''.join(f'{chrom}\t{start}\t{w}\t{m}\t{posterior}\t{state_names[state]}\t{var}\t{archaic_variants_by_position(chrom, var)}\n'
                              for chrom, start, w, m, posterior, state, var in rows))



def Convert_genome_coordinates(window_size, CHROMOSOME_BREAKPOINTS, starts, variants, post_seq, path, hmm_parameters, weights, mutrates, obs):

    segments = []
    for (chrom, chrom_start_index, chrom_length_index) in CHROMOSOME_BREAKPOINTS:

        # Diploid or haploid
        if '_hap' in chrom:
            newchrom, ploidity = chrom.split('_')
        else:
            ploidity = 'diploid'
            newchrom = chrom

        state_with_highest_prob = path[chrom_start_index:chrom_start_index + chrom_length_index]

        for (state, start_index, length_index) in find_runs(state_with_highest_prob):
            
            start_index = start_index + chrom_start_index
            end_index = start_index + length_index

            genome_start = starts[start_index]
            genome_length =  length_index * window_size
            genome_end = genome_start + genome_length

            called_sequence = int(np.sum(weights[start_index:end_index]) * window_size)
            average_mutation_rate = round(np.mean(mutrates[start_index:end_index]), 3)

            snp_counter = np.sum(obs[start_index:end_index])
            mean_prob = round(np.mean(post_seq[state, start_index:end_index]), 5)
            variants_segment = flatten_list(variants[start_index:end_index])

            segments.append([newchrom, 
                             genome_start,  
                             genome_end, 
                             genome_length, 
                             hmm_parameters.state_names[state], 
                             mean_prob, 
                             snp_counter, 
                             ploidity, 
                             called_sequence, 
                             average_mutation_rate, 
                             variants_segment]) 
    
    return segments



def Write_Decoded_output(outputprefix, segments, obs_file, admixpop_file, extrainfo):

    # Load archaic data
    if admixpop_file is not None:
        admix_pop_variants, admixpop_names = Annotate_with_ref_genome(admixpop_file, obs_file)

    # Are we doing haploid/diploid?
    outfile_mapper = {}
    for _, _, _, _, _, _, _, ploidity, _, _, _ in segments:
        if outputprefix == '/dev/stdout':
            outfile_mapper[ploidity] = '/dev/stdout'
        else:
            outfile_mapper[ploidity] = f'{outputprefix}.{ploidity}.txt'


    # Make output files and write headers
    outputfiles_handlers = defaultdict(str)
    for ploidity, output in outfile_mapper.items():
       
        Make_folder_if_not_exists(output)
        outputfiles_handlers[ploidity] = open(output, 'w')
        out = outputfiles_handlers[ploidity]

        # Make header
        HEADER_COLUMNS = ['chrom','start','end','length','state','mean_prob','snps']
        if admixpop_file is not None:
            HEADER_COLUMNS += ['admixpopvariants', '\t'.join(admixpop_names)]
            if extrainfo:
                HEADER_COLUMNS += ['called_sequence','mutationrate','variants','DAV_variants']
        else:
            if extrainfo:
                HEADER_COLUMNS += ['called_sequence','mutationrate','variants']
        print('\t'.join(HEADER_COLUMNS), file = out)

    # Go through segments and write to output
    for chrom, genome_start, genome_end, genome_length, state, mean_prob, snp_counter, ploidity, called_sequence, average_mutation_rate, variants  in segments:

        out = outputfiles_handlers[ploidity]

        if admixpop_file is not None:
            archiac_variants_dict = defaultdict(int)
            archaic_variants_by_position = []
            
            for snp_position in variants.split(','):
                carriers = admix_pop_variants[f'{chrom}_{snp_position}']
                archaic_variants_by_position.append(carriers)

                if carriers != 'none':
                    for ind in carriers.split('|'):
                        archiac_variants_dict[ind] += 1
                    archiac_variants_dict['total'] += 1
                
            archaic_variants = '\t'.join([str(archiac_variants_dict[x]) for x in ['total'] + admixpop_names])
            archaic_variants_by_position = ','.join(archaic_variants_by_position)

            if extrainfo:
                print(chrom, genome_start, genome_end, genome_length, state, mean_prob, snp_counter, archaic_variants, called_sequence, average_mutation_rate, variants, archaic_variants_by_position, sep = '\t', file = out)
            else:
                print(chrom, genome_start, genome_end, genome_length, state, mean_prob, snp_counter, archaic_variants, sep = '\t', file = out)

        else:

            if extrainfo:
                print(chrom, genome_start, genome_end, genome_length, state, mean_prob, snp_counter, called_sequence, average_mutation_rate, variants, sep = '\t', file = out)
            else:    
                print(chrom, genome_start, genome_end, genome_length, state, mean_prob, snp_counter, sep = '\t', file = out)


    # Close output files
    for ploidity, out in outputfiles_handlers.items():
        out.close()



