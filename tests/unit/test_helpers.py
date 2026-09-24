"""Tests for the input handling helpers.

Some tests pin down current behaviour that may not be intended (marked
"characterisation"): they exist so that refactoring cannot change results
silently, and should be updated deliberately if the behaviour is changed.
"""

import numpy as np
import pytest

from hmmix.helper_functions import (
    Load_observations_weights_mutrates,
    combined_files,
    convert_to_bases,
    find_runs,
    flatten_list,
    get_consensus,
    handle_individuals_input,
    handle_infiles,
    load_fasta,
    make_callability_from_bed,
    sortby,
    sortby_haplotype,
)
from hmmix.make_mutationrate import make_mutation_rate

OBS_HEADER = 'chrom\tpos\tancestral_base\tgenotype\n'


def write(path, text):
    path.write_text(text)
    return str(path)


def test_find_runs():
    assert list(find_runs([0, 0, 1, 1, 1, 0])) == [(0, 0, 2), (1, 2, 3), (0, 5, 1)]
    assert list(find_runs(['chr1', 'chr1', 'chr2'])) == [('chr1', 0, 2), ('chr2', 2, 1)]
    assert list(find_runs([5])) == [(5, 0, 1)]


def test_sortby_orders_numbers_then_letters():
    assert sorted(['X', '10', '2', 'MT', '1', 'Y'], key=sortby) == ['1', '2', '10', 'MT', 'X', 'Y']
    assert sorted(['chr10', 'chr2'], key=sortby) == ['chr10', 'chr2']  # all 'c' - stable
    assert sorted(['_hap2', '_hap10', '_hap1'], key=sortby_haplotype) == ['_hap1', '_hap2', '_hap10']


def test_flatten_list_skips_empty():
    assert flatten_list(['1,2', '', '5']) == '1,2,5'
    assert flatten_list(['', '']) == ''


@pytest.mark.parametrize('genotype, expected', [
    ('0/1', 'AT'), ('1|1', 'TT'), ('0|2', 'AC'), ('./.', 'NN'), ('0', 'NN'), ('0/3', 'NN'),
])
def test_convert_to_bases(genotype, expected):
    assert convert_to_bases(genotype, ['A', 'T', 'C', '*']) == expected


def test_get_consensus():
    prefix, suffix, values = get_consensus(['chr1.vcf', 'chr10.vcf', 'chr2.vcf'])
    assert (prefix, suffix, values) == ('chr', '.vcf', ['1', '2', '10'])
    assert get_consensus(['only.vcf']) == (None, None, None)


def test_handle_infiles(tmp_path):
    for c in ['1', '2']:
        (tmp_path / f'chr{c}.vcf').write_text('')
    assert sorted(handle_infiles(str(tmp_path / 'chr*.vcf'))) == [str(tmp_path / 'chr1.vcf'), str(tmp_path / 'chr2.vcf')]
    assert handle_infiles('a.vcf,b.vcf') == ['a.vcf', 'b.vcf']


def test_combined_files():
    assert combined_files([''], ['x.vcf']) == ([None], ['x.vcf'])
    anc, vcf = combined_files(['anc_2.fa', 'anc_1.fa', 'anc_3.fa'], ['c1.vcf', 'c2.vcf'])
    assert (anc, vcf) == (['anc_1.fa', 'anc_2.fa'], ['c1.vcf', 'c2.vcf'])


def test_handle_individuals_input(tmp_path):
    f = write(tmp_path / 'ind.json', '{"ingroup": ["a", "b"], "outgroup": ["c"]}')
    assert handle_individuals_input(f, 'outgroup') == ['c']
    assert handle_individuals_input('x,y', 'ingroup') == ['x', 'y']


def test_load_fasta(tmp_path):
    f = write(tmp_path / 'a.fa', '>chr1\nacgt\nNNAC\n')
    assert load_fasta(f) == 'ACGTNNAC'


def test_callability_characterisation(tmp_path):
    # characterisation: BED end coordinates are treated as inclusive, so [0, 1000)
    # adds one base to window 1000 and [5000, 5100) counts 101 bases.
    bed = write(tmp_path / 'w.bed', 'chr1\t0\t1000\nchr1\t1500\t3200\nchr1\t5000\t5100\t0.5\nchr2\t0\t2500\n')
    call = make_callability_from_bed(bed, 1000)
    assert dict(call['chr1']) == {0: 1000.0, 1000: 501.0, 2000: 1000.0, 3000: 201.0, 5000: 50.5}
    assert dict(call['chr2']) == {0: 1000.0, 1000: 1000.0, 2000: 501.0}


def test_callability_skips_header(tmp_path):
    bed = write(tmp_path / 'w.bed', 'chrom\tstart\tend\nchr1\t10\t20\n')
    assert dict(make_callability_from_bed(bed, 1000)['chr1']) == {0: 11.0}


@pytest.fixture
def small_obs(tmp_path):
    return write(tmp_path / 'obs.txt', OBS_HEADER + 'chr1\t10\tA\tAG\nchr1\t2500\tC\tCT\nchr1\t2600\tC\tTT\nchr1\t4999\tG\tGA\nchr2\t1500\tT\tTA\nchr2\t3100\tT\tTA\n')


def test_load_observations_defaults(small_obs):
    obs, chroms, starts, variants, mutrates, weights = Load_observations_weights_mutrates(small_obs, None, None, 1000, False, 'All')
    # characterisation: the window holding the last SNP of a chromosome is not included
    assert chroms == ['chr1'] * 4 + ['chr2'] * 3
    assert starts.tolist() == [0, 1000, 2000, 3000, 0, 1000, 2000]
    assert obs.tolist() == [1, 0, 2, 0, 0, 1, 0]
    assert variants == ['10', '', '2500,2600', '', '', '1500', '']
    assert obs.dtype.kind == 'i' and weights.dtype == float and mutrates.dtype == float
    np.testing.assert_array_equal(weights, 1.0)
    np.testing.assert_array_equal(mutrates, 1.0)


def test_load_observations_haploid_and_subset(small_obs):
    obs, chroms, starts, variants, _, _ = Load_observations_weights_mutrates(small_obs, None, None, 1000, True, 'chr1')
    assert chroms == ['chr1_hap1'] * 4 + ['chr1_hap2'] * 4
    assert obs.tolist() == [0, 0, 1, 0, 1, 0, 2, 0]
    assert variants == ['', '', '2600', '', '10', '', '2500,2600', '']


def test_load_observations_weights_and_mutrates(tmp_path, small_obs, capsys):
    weights = write(tmp_path / 'w.bed', 'chr1\t0\t2999\nchr2\t0\t1999\n')
    mutrates = write(tmp_path / 'm.bed', 'chr1\t0\t5999\t2\nchr2\t0\t1999\t0.5\n')
    obs, chroms, starts, _, mut, w = Load_observations_weights_mutrates(small_obs, weights, mutrates, 1000, False, 'All')
    # spans come from the weights file when given
    assert chroms == ['chr1'] * 2 + ['chr2']
    assert w.tolist() == [1.0, 1.0, 1.0]
    assert mut.tolist() == [2.0, 2.0, 0.5]
    # characterisation: chr2 ends in window 1000, which is excluded, so the SNP at 1500 is lost
    assert obs.tolist() == [1, 0, 0]


def test_load_observations_zeroes_uncallable(tmp_path, capsys):
    obs_file = write(tmp_path / 'obs.txt', OBS_HEADER + 'chr1\t1500\tA\tAG\nchr1\t5500\tA\tAG\n')
    weights = write(tmp_path / 'w.bed', 'chr1\t0\t999\nchr1\t2000\t4999\n')
    obs, *_ , w = Load_observations_weights_mutrates(obs_file, weights, None, 1000, False, 'All')
    assert w.tolist() == [1.0, 0.0, 1.0, 1.0]
    assert obs.tolist() == [0, 0, 0, 0]
    assert 'no called bases' in capsys.readouterr().out


def test_load_observations_empty(tmp_path):
    obs_file = write(tmp_path / 'obs.txt', OBS_HEADER)
    weights = write(tmp_path / 'w.bed', 'chr1\t0\t2999\n')
    obs, chroms, *_ = Load_observations_weights_mutrates(obs_file, weights, None, 1000, False, 'All')
    assert obs.tolist() == [0, 0] and chroms == ['chr1', 'chr1']
    with pytest.raises(SystemExit):
        Load_observations_weights_mutrates(obs_file, weights, None, 1000, True, 'All')


def test_make_mutation_rate(tmp_path):
    outgroup = write(tmp_path / 'out.txt', 'chrom\tpos\n' + ''.join(f'chr1\t{p}\n' for p in [5, 10, 15, 1500, 2500, 2600]))
    out = tmp_path / 'sub' / 'rates.bed'
    make_mutation_rate(outgroup, str(out), None, 1000)
    lines = out.read_text().splitlines()
    assert lines[0] == 'chrom\tstart\tend\tmutationrate'
    # 6 snps over 3 windows -> mean 2 per window
    assert [line.split('\t') for line in lines[1:]] == [['chr1', '0', '1000', '1.5'], ['chr1', '1000', '2000', '0.5'], ['chr1', '2000', '3000', '1.0']]
