"""The optimised loader must return exactly what the 0.9.2 loader returned."""

import random

import numpy as np
import pytest

from hmmix.helper_functions import Load_observations_weights_mutrates, callability_arrays_from_bed, make_callability_from_bed

from .reference_loader import Load_observations_weights_mutrates as reference_loader

CHROMS = ['1', '2', '10', 'X', 'chr3']


def random_bed(rng, path, with_values):
    lines = []
    for chrom in rng.sample(CHROMS, rng.randint(1, len(CHROMS))):
        for _ in range(rng.randint(1, 6)):
            start = rng.randint(0, 40_000)
            end = start + rng.choice([0, 1, 50, 999, 1000, 1001, rng.randint(0, 20_000)])
            if with_values:
                lines.append(f'{chrom}\t{start}\t{end}\t{rng.choice([0, 0.3, 1, 1.7, 2.25])}')
            else:
                lines.append(f'{chrom}\t{start}\t{end}')
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


def random_obs(rng, path):
    lines = ['chrom\tpos\tancestral_base\tgenotype']
    for _ in range(rng.randint(0, 60)):
        anc = rng.choice('ACGT')
        der = rng.choice([b for b in 'ACGT' if b != anc])
        genotype = rng.choice([anc + der, der + anc, der + der])
        lines.append(f'{rng.choice(CHROMS)}\t{rng.randint(1, 45_000)}\t{anc}\t{genotype}')
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


def assert_same(a, b):
    obs_a, chroms_a, starts_a, variants_a, mut_a, w_a = a
    obs_b, chroms_b, starts_b, variants_b, mut_b, w_b = b
    np.testing.assert_array_equal(obs_a, obs_b)
    assert obs_a.dtype == obs_b.dtype
    assert list(chroms_a) == list(chroms_b)
    assert [int(x) for x in starts_a] == [int(x) for x in starts_b]
    assert list(variants_a) == list(variants_b)
    # bitwise identical floating point values
    assert mut_a.tobytes() == mut_b.tobytes()
    assert w_a.tobytes() == w_b.tobytes()


@pytest.mark.parametrize('seed', range(200))
def test_loader_matches_reference(tmp_path, seed, capsys):
    rng = random.Random(seed)
    obs = random_obs(rng, tmp_path / 'obs.txt')
    weights = random_bed(rng, tmp_path / 'w.bed', with_values=rng.random() < 0.3) if rng.random() < 0.7 else None
    mutrates = random_bed(rng, tmp_path / 'm.bed', with_values=True) if rng.random() < 0.7 else None
    window_size = rng.choice([100, 1000, 1500])
    haploid = rng.random() < 0.4
    chrom = rng.choice(['All', 'All', '2', '1,X', 'chr3,10'])
    args = (obs, weights, mutrates, window_size, haploid, chrom)

    try:
        expected = reference_loader(*args)
    except SystemExit:
        with pytest.raises(SystemExit):
            Load_observations_weights_mutrates(*args)
        return
    expected_log = capsys.readouterr().out
    actual = Load_observations_weights_mutrates(*args)
    assert capsys.readouterr().out == expected_log
    assert_same(actual, expected)


@pytest.mark.parametrize('seed', range(50))
def test_callability_arrays_match_dict_version(tmp_path, seed):
    rng = random.Random(seed)
    bed = random_bed(rng, tmp_path / 'b.bed', with_values=rng.random() < 0.5)
    window_size = rng.choice([100, 1000])
    expected = make_callability_from_bed(bed, window_size)
    actual = callability_arrays_from_bed(bed, window_size)
    assert list(actual) == list(expected)
    for chrom, windows in expected.items():
        assert actual[chrom].last_window == max(windows)
        dense = actual[chrom].values
        for window, value in windows.items():
            assert dense[window // window_size] == value
        assert np.count_nonzero(dense) == sum(1 for v in windows.values() if v != 0)
