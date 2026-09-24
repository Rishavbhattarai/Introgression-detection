"""Regression tests: the current code must reproduce the outputs of hmmix 0.9.2.

The golden files were produced by running run_pipeline.sh against the unmodified
0.9.2 release. Every output file is compared token by token; numeric tokens are
allowed a tiny relative difference so that the test is not sensitive to
last-digit floating point differences between platforms / BLAS builds.
"""

import gzip
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
GOLDEN = HERE / 'golden'
REL_TOL = 1e-9

# Files that are produced by the pipeline and compared against the golden copies.
GOLDEN_FILES = sorted(p.name[:-3] if p.suffix == '.gz' else p.name for p in GOLDEN.iterdir())


def read_text(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as fh:
            return fh.read()
    return path.read_text()


def tokens_equal(a, b):
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    return math.isclose(x, y, rel_tol=REL_TOL, abs_tol=1e-12)


def assert_same_text(expected, actual, name):
    exp_lines = expected.splitlines()
    act_lines = actual.splitlines()
    assert len(exp_lines) == len(act_lines), f'{name}: {len(act_lines)} lines, expected {len(exp_lines)}'
    for lineno, (e, a) in enumerate(zip(exp_lines, act_lines), 1):
        e_tok, a_tok = e.replace(',', ' , ').split(), a.replace(',', ' , ').split()
        ok = len(e_tok) == len(a_tok) and all(tokens_equal(x, y) for x, y in zip(e_tok, a_tok))
        assert ok, f'{name}:{lineno}\n expected: {e}\n actual:   {a}'


@pytest.fixture(scope='module')
def pipeline_output(tmp_path_factory):
    out = tmp_path_factory.mktemp('pipeline')
    cmd = ['bash', str(HERE / 'run_pipeline.sh'), str(out), sys.executable, '-m', 'hmmix']
    subprocess.run(cmd, check=True, env={**os.environ, 'MPLBACKEND': 'Agg'})
    return out


@pytest.mark.slow
@pytest.mark.parametrize('name', GOLDEN_FILES)
def test_output_matches_golden(pipeline_output, name):
    golden = GOLDEN / name
    if not golden.exists():
        golden = GOLDEN / (name + '.gz')
    produced = pipeline_output / name
    assert produced.exists(), f'{name} was not produced'
    assert_same_text(read_text(golden), produced.read_text(), name)
