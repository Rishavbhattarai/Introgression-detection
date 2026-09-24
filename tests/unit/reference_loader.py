"""The observation loader from hmmix 0.9.2, kept verbatim as a reference implementation
for tests/unit/test_loader_equivalence.py."""
# ruff: noqa
import sys
from collections import defaultdict

import numpy as np

from hmmix.helper_functions import make_callability_from_bed, sortby, sortby_haplotype


def Load_observations_weights_mutrates(obs_file, weights_file, mutrates_file, window_size, haploid, chrom_to_look_for):
    '''
    First get structure of genome
    1) read obs file > ploidity, chrom number and lengths
    2) overwrite lengths with weights
    3) only keep chromosomes we want
    4) fill in obs
    5) fill in weights and mutation rates
    6) make sure no invalid windows exists 

    '''

    # get span of observation data
    chromosome_spans = defaultdict(int)
    with open(obs_file) as data:
        for line in data:
            if line.startswith('chrom'):
                continue
            chrom, pos, ancestral_base, genotype = line.strip().split()

            # convert 1-indexed position to 0-indexed position
            zero_based_pos = int(pos) - 1
            rounded_pos = zero_based_pos - zero_based_pos % window_size
            chromosome_spans[chrom] = rounded_pos


    # get span of callability
    if weights_file: 
        callability = make_callability_from_bed(weights_file, window_size)
        for chrom in callability:
            chromosome_spans[chrom] = max(callability[chrom])

    # which chromosomes are we interested in
    if chrom_to_look_for != 'All':
        chromosome_list = sorted(chrom_to_look_for.split(','), key=sortby)
    else:
        chromosome_list = sorted(list(chromosome_spans.keys()), key=sortby)

    obs_counter = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    haplotypes = defaultdict(int)
    chroms, starts, variants, obs = [], [], [], []

    # read observation data
    with open(obs_file) as data:
        for line in data:
            if line.startswith('chrom'):
                continue 

            chrom, pos, ancestral_base, genotype = line.strip().split()
            if not chrom in chromosome_list:
                continue

            # convert 1-indexed position to 0-indexed position
            zero_based_pos = int(pos) - 1
            rounded_pos = zero_based_pos - zero_based_pos % window_size

            if haploid:  
                for i, base in enumerate(genotype):
                    if base != ancestral_base:
                        obs_counter[chrom][rounded_pos][f'_hap{i+1}'].append(pos)
                        haplotypes[f'_hap{i+1}'] += 1
            else:
                obs_counter[chrom][rounded_pos][''].append(pos)
                haplotypes[''] += 1


    # take care of cases where there is 0 observations
    if len(haplotypes) == 0:
        if haploid:
            sys.exit('Could not determine haploidity because there is no data')
        else:
            haplotypes[''] += 1
        

    for haplotype in sorted(haplotypes, key=sortby_haplotype):
        
        for chrom in chromosome_list:
            lastwindow = chromosome_spans[chrom]

            for window in range(0, lastwindow, window_size):
                chroms.append(f'{chrom}{haplotype}')   
                starts.append(window)
                variants.append(','.join(obs_counter[chrom][window][haplotype]))  
                obs.append(len(obs_counter[chrom][window][haplotype])) 
            

    # Read weights file is it exists - else set all weights to 1
    if weights_file is None:
        weights = np.ones(len(obs)) 
    else:  
        callability = make_callability_from_bed(weights_file, window_size)
        weights = []
        for haplotype in sorted(haplotypes, key=sortby_haplotype):
            
            for chrom in chromosome_list:
                lastwindow = chromosome_spans[chrom]

                for window in range(0, lastwindow, window_size):
                    weights.append(callability[chrom][window] / float(window_size))


    # Read mutation rate file is it exists - else set all mutation rates to 1
    if mutrates_file is None:
        mutrates = np.ones(len(obs)) 
    else:  
        callability = make_callability_from_bed(mutrates_file, window_size)
        mutrates = []
        for haplotype in sorted(haplotypes, key=sortby_haplotype):
            
            for chrom in chromosome_list:
                lastwindow = chromosome_spans[chrom]

                for window in range(0, lastwindow, window_size):
                    mutrates.append(callability[chrom][window] / float(window_size))



    # Make sure there are no places with obs > 0 and 0 in mutation rate or weight
    for index, (observation, w, m) in enumerate(zip(obs, weights, mutrates)):
        if w*m == 0 and observation != 0:
            print(f'warning, you had {observation} observations but no called bases/no mutation rate at index:{index}. weights:{w}, mutrates:{m}')
            obs[index] = 0
            
    return np.array(obs).astype(int), chroms, starts, variants, np.array(mutrates).astype(float), np.array(weights).astype(float)
