"""Turn the JSON written by run_benchmarks.py into markdown tables and a scaling plot.

    python benchmarks/make_report.py benchmark_results.json --plot benchmarks/scaling.png
"""

import argparse
import json


def fit_slope(xs, ys):
    """Least squares slope of ys against xs"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def tables(results):
    lines = []
    for size in results['sizes']:
        lines.append(f"**{size['windows']:,} windows ({size['snps']:,} SNPs)**\n")
        lines.append('| command | 0.9.2 time | new time | speed-up | 0.9.2 peak memory | new peak memory | memory reduction | same output |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|:---:|')
        for name, entry in size['commands'].items():
            b, c = entry['baseline'], entry['candidate']
            same = 'yes' if entry['outputs_identical'] else 'no'
            if 'max_rel_diff_trained_parameters' in entry:
                same = f"≤{entry['max_rel_diff_trained_parameters']:.0e} rel."
            lines.append(f"| {name} | {b['wall_s']:.2f} s | {c['wall_s']:.2f} s | {entry['speedup']:.1f}× | "
                         f"{b['max_rss_mb']:,.0f} MB | {c['max_rss_mb']:,.0f} MB | {entry['memory_reduction']:.1f}× | {same} |")
        lines.append('')

    if len(results['sizes']) > 1:
        lines.append('**Memory growth per window** (slope of peak memory against number of windows)\n')
        lines.append('| command | 0.9.2 | new |')
        lines.append('|---|---:|---:|')
        windows = [s['windows'] for s in results['sizes']]
        for name in results['sizes'][0]['commands']:
            slopes = []
            for tag in ['baseline', 'candidate']:
                mem = [s['commands'][name][tag]['max_rss_mb'] * 2**20 for s in results['sizes']]
                slopes.append(fit_slope(windows, mem))
            lines.append(f'| {name} | {slopes[0]:.0f} bytes | {slopes[1]:.0f} bytes |')
        lines.append('')
    return '\n'.join(lines)


def plot(results, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    windows = [s['windows'] / 1e6 for s in results['sizes']]
    names = list(results['sizes'][0]['commands'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, name in enumerate(names):
        color = f'C{i}'
        for tag, style in [('baseline', '--'), ('candidate', '-')]:
            label = f'{name} ({"0.9.2" if tag == "baseline" else "new"})'
            axes[0].plot(windows, [s['commands'][name][tag]['wall_s'] for s in results['sizes']], style, color=color, marker='o', label=label)
            axes[1].plot(windows, [s['commands'][name][tag]['max_rss_mb'] / 1024 for s in results['sizes']], style, color=color, marker='o')
    axes[0].set(xlabel='windows (millions)', ylabel='wall time (s)', title='Run time')
    axes[1].set(xlabel='windows (millions)', ylabel='peak memory (GB)', title='Peak memory')
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle('hmmix 0.9.2 (dashed) vs this branch (solid)')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('--plot', help='write a scaling plot to this file')
    args = parser.parse_args()
    results = json.loads(open(args.results).read())
    m = results['machine']
    print(f"Machine: {m['platform']}, {m['cpus']} CPUs, Python {m['python']}\n")
    print(tables(results))
    if args.plot:
        plot(results, args.plot)


if __name__ == '__main__':
    main()
