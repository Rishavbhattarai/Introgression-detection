# Benchmarks

hmmix 0.9.2 (the PyPI release) compared with this branch on simulated data: 10 chromosomes of
1 kb windows, generated with `hmmix make_test_data` using a 2-state model with frequent archaic
tracts. 3 million windows is about the size of one diploid human genome at the default window size.

Reproduce with

```bash
python benchmarks/run_benchmarks.py --baseline <env-0.9.2>/bin/hmmix --candidate <env-new>/bin/hmmix \
    --windows 30000 100000 300000 --out benchmarks/results.json
python benchmarks/make_report.py benchmarks/results.json --plot benchmarks/scaling.png
```

Times are the median of 3 runs after one warm-up run; memory is the peak resident set size of the
hmmix process. Both versions used Python 3.13, numpy 2.5.3 and numba 0.67.0 (raw numbers:
[results.json](results.json)). "same output" compares the output files byte by byte; for `train`
the largest relative difference between the trained parameters is given instead.

## Summary

At 3 million windows every subcommand is **6.0–6.7× faster** and peak memory drops from about
**2 GB to 0.6–0.9 GB**, with the same output. Memory per window goes from ~620 to ~150–220 bytes.
Memory still grows with the number of windows: forward–backward over the concatenated genome
needs arrays proportional to its length, so this is a constant-factor reduction.

The first run of a new installation compiles the numba functions (~3 s) and caches them;
0.9.2 paid that cost on every command, which is why small inputs speed up the most (~8–9×).

![scaling](scaling.png)

## Where the time and memory went (0.9.2, 1M windows)

| cost | fix |
|---|---|
| numba recompiled every function on every command (~3 s) | `@njit(cache=True)` |
| loader made a dict + list for every window (~500 bytes/window, 0.5 GB) | numpy arrays built directly (~50 bytes/window) |
| posterior table written one rounded value at a time (most of `decode`) | vectorised rounding, chunked writes |
| training ran the forward algorithm twice per iteration | forward pass reused |
| numba called BLAS for each 2×2 `@` on every window | explicit loops (also removes the scipy dependency) |
| Python loops over windows in simulation / sampling / normalisation | numba loops with identical random draws; vectorised normalisation |

## Results

**300,000 windows (22,177 SNPs)**

| command | 0.9.2 time | new time | speed-up | 0.9.2 peak memory | new peak memory | memory reduction | same output |
|---|---:|---:|---:|---:|---:|---:|:---:|
| train | 3.20 s | 0.42 s | 7.6× | 392 MB | 188 MB | 2.1× | ≤6e-14 rel. |
| decode | 3.28 s | 0.49 s | 6.6× | 365 MB | 206 MB | 1.8× | yes |
| decode_viterbi | 2.40 s | 0.31 s | 7.8× | 363 MB | 160 MB | 2.3× | yes |
| inhomogeneous | 4.11 s | 0.56 s | 7.4× | 387 MB | 203 MB | 1.9× | yes |
| artemis | 3.72 s | 0.84 s | 4.4× | 621 MB | 414 MB | 1.5× | yes |

**1,000,000 windows (73,749 SNPs)**

| command | 0.9.2 time | new time | speed-up | 0.9.2 peak memory | new peak memory | memory reduction | same output |
|---|---:|---:|---:|---:|---:|---:|:---:|
| train | 7.58 s | 1.03 s | 7.4× | 940 MB | 299 MB | 3.1× | ≤8e-15 rel. |
| decode | 7.36 s | 1.17 s | 6.3× | 890 MB | 364 MB | 2.4× | yes |
| decode_viterbi | 3.74 s | 0.53 s | 7.0× | 926 MB | 292 MB | 3.2× | yes |
| inhomogeneous | 9.21 s | 1.39 s | 6.6× | 986 MB | 421 MB | 2.3× | yes |
| artemis | 6.60 s | 1.24 s | 5.3× | 1,164 MB | 585 MB | 2.0× | yes |

**3,000,000 windows (217,985 SNPs)**

| command | 0.9.2 time | new time | speed-up | 0.9.2 peak memory | new peak memory | memory reduction | same output |
|---|---:|---:|---:|---:|---:|---:|:---:|
| train | 20.35 s | 3.02 s | 6.7× | 2,033 MB | 581 MB | 3.5× | ≤2e-13 rel. |
| decode | 18.97 s | 3.16 s | 6.0× | 2,038 MB | 690 MB | 3.0× | yes |
| decode_viterbi | 7.51 s | 1.21 s | 6.2× | 2,035 MB | 607 MB | 3.4× | yes |
| inhomogeneous | 23.74 s | 3.75 s | 6.3× | 2,035 MB | 797 MB | 2.6× | yes |
| artemis | 14.96 s | 2.35 s | 6.4× | 2,140 MB | 859 MB | 2.5× | yes |

**Memory growth per window** (slope of peak memory against number of windows)

| command | 0.9.2 | new |
|---|---:|---:|
| train | 623 bytes | 151 bytes |
| decode | 639 bytes | 184 bytes |
| decode_viterbi | 634 bytes | 172 bytes |
| inhomogeneous | 620 bytes | 223 bytes |
| artemis | 573 bytes | 166 bytes |

