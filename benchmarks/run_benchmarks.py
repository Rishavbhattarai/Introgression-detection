"""Benchmark two hmmix installations against each other.

    python benchmarks/run_benchmarks.py \
        --baseline /path/to/old/env/bin/hmmix --candidate /path/to/new/env/bin/hmmix \
        --windows 30000 100000 300000 --out results.json

For every data size, test data is simulated (10 chromosomes of --windows 1 kb windows),
then each subcommand is run --repeats times with each installation. Wall time is the
median, memory is the peak resident set size of the hmmix process. Before the timed
runs each command is run once, so the candidate's numba cache exists; the time of
that first run is reported as 'first_run_s'. Output files of the two installations
are compared and the result is recorded.
"""

import argparse
import filecmp
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

SIM_PARAMS = {
    'state_names': ['Human', 'Archaic'],
    'starting_probabilities': [0.9, 0.1],
    'transitions': [[0.995, 0.005], [0.05, 0.95]],
    'emissions': [0.04, 0.4],
}

COMMANDS = {
    'train': ['train', '-obs=obs.txt', '-weights=weights.bed', '-mutrates=mutrates.bed', '-param=Initialguesses.json', '-out={tag}.trained.json'],
    'decode': ['decode', '-obs=obs.txt', '-weights=weights.bed', '-mutrates=mutrates.bed', '-param=sim_params.json', '-out={tag}.decoded', '-extrainfo', '-posterior_probs={tag}.posterior.txt'],
    'decode_viterbi': ['decode', '-obs=obs.txt', '-weights=weights.bed', '-mutrates=mutrates.bed', '-param=sim_params.json', '-out={tag}.viterbi', '-viterbi'],
    'inhomogeneous': ['inhomogeneous', '-obs=obs.txt', '-weights=weights.bed', '-mutrates=mutrates.bed', '-param=sim_params.json', '-out={tag}.inhom', '-samples=10'],
    'artemis': ['artemis', '-obs=obs.txt', '-weights=weights.bed', '-mutrates=mutrates.bed', '-param=sim_params.json', '-out={tag}.artemis.txt', '-out_plot={tag}.artemis.png', '-steps=21'],
}

# outputs that must be identical between the two installations (trained parameters are
# compared numerically instead, see compare_trained)
IDENTICAL_OUTPUTS = {
    'decode': ['{tag}.decoded.diploid.txt', '{tag}.posterior.txt'],
    'decode_viterbi': ['{tag}.viterbi.diploid.txt'],
    'inhomogeneous': ['{tag}.inhom.0.diploid.txt', '{tag}.inhom.9.diploid.txt'],
    'artemis': ['{tag}.artemis.txt'],
}


def run(cmd, cwd):
    """Run cmd, return (wall seconds, peak RSS in MB) of that process"""
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env={**os.environ, 'MPLBACKEND': 'Agg'})
    _, status, usage = os.wait4(proc.pid, 0)
    wall = time.perf_counter() - start
    stderr = proc.stderr.read().decode()
    proc.stderr.close()
    if os.waitstatus_to_exitcode(status) != 0:
        raise RuntimeError(f'{cmd} failed:\n{stderr}')
    # ru_maxrss is in bytes on macOS and kilobytes on Linux
    rss = usage.ru_maxrss / (2**20 if sys.platform == 'darwin' else 2**10)
    return wall, rss


def compare_trained(a, b):
    """Largest relative difference between two trained parameter files"""
    def flat(x):
        return [v for item in x for v in (flat(item) if isinstance(item, list) else [item])]
    pa, pb = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    diffs = [abs(x - y) / abs(x) for key in ['starting_probabilities', 'transitions', 'emissions']
             for x, y in zip(flat(pa[key]), flat(pb[key])) if x != 0]
    return max(diffs)


def benchmark_size(windows, args, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / 'sim_params.json').write_text(json.dumps(SIM_PARAMS))
    wall, rss = run([args.candidate, 'make_test_data', f'-windows={windows}', f'-chromosomes={args.chromosomes}', '-param=sim_params.json'], workdir)
    n_windows = windows * args.chromosomes
    n_snps = sum(1 for _ in open(workdir / 'obs.txt')) - 1
    print(f'\n== {n_windows:,} windows, {n_snps:,} SNPs', flush=True)

    result = {'windows': n_windows, 'snps': n_snps, 'commands': {}}
    for name, template in COMMANDS.items():
        if name not in args.commands:
            continue
        entry = {}
        for tag, exe in [('baseline', args.baseline), ('candidate', args.candidate)]:
            cmd = [exe] + [part.format(tag=tag) for part in template]
            first_wall, _ = run(cmd, workdir)
            walls, rsss = zip(*(run(cmd, workdir) for _ in range(args.repeats)))
            entry[tag] = {'first_run_s': first_wall, 'wall_s': statistics.median(walls), 'wall_all_s': list(walls), 'max_rss_mb': max(rsss)}
        entry['speedup'] = entry['baseline']['wall_s'] / entry['candidate']['wall_s']
        entry['memory_reduction'] = entry['baseline']['max_rss_mb'] / entry['candidate']['max_rss_mb']

        same = [filecmp.cmp(workdir / f.format(tag='baseline'), workdir / f.format(tag='candidate'), shallow=False) for f in IDENTICAL_OUTPUTS.get(name, [])]
        entry['outputs_identical'] = all(same)
        if name == 'train':
            entry['max_rel_diff_trained_parameters'] = compare_trained(workdir / 'baseline.trained.json', workdir / 'candidate.trained.json')

        result['commands'][name] = entry
        b, c = entry['baseline'], entry['candidate']
        print(f"{name:15s} {b['wall_s']:7.2f}s -> {c['wall_s']:6.2f}s ({entry['speedup']:4.1f}x)   "
              f"{b['max_rss_mb']:6.0f} MB -> {c['max_rss_mb']:5.0f} MB ({entry['memory_reduction']:3.1f}x)   identical={entry['outputs_identical']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', required=True, help='hmmix executable of the reference installation')
    parser.add_argument('--candidate', required=True, help='hmmix executable to compare')
    parser.add_argument('--windows', type=int, nargs='+', default=[30_000, 100_000], help='1 kb windows per chromosome')
    parser.add_argument('--chromosomes', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--commands', nargs='+', default=list(COMMANDS), choices=list(COMMANDS))
    parser.add_argument('--workdir', default='benchmark_data')
    parser.add_argument('--out', default='benchmark_results.json')
    args = parser.parse_args()

    results = {
        'machine': {'platform': platform.platform(), 'processor': platform.processor(), 'python': platform.python_version(), 'cpus': os.cpu_count()},
        'sizes': [],
    }
    for windows in args.windows:
        workdir = Path(args.workdir) / f'w{windows}'
        results['sizes'].append(benchmark_size(windows, args, workdir))
        shutil.rmtree(workdir)
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
