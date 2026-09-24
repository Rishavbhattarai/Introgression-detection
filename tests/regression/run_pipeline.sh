#!/usr/bin/env bash
# Run every hmmix subcommand that does not need bcftools on small, fixed inputs.
# Usage: run_pipeline.sh <outdir> <hmmix command...>
# The outputs of the original 0.9.2 code are stored in tests/regression/golden and
# test_regression.py checks that the current code reproduces them.
set -euo pipefail
OUT=$1; shift
HMMIX=("$@")
mkdir -p "$OUT"; cd "$OUT"

# 1) simulated tutorial data (obs.txt, weights.bed, mutrates.bed, Initialguesses.json)
# Parameters with frequent archaic tracts so decoding has something to find.
cat > sim_params.json <<'JSON'
{"state_names": ["Human", "Archaic"], "starting_probabilities": [0.9, 0.1],
 "transitions": [[0.995, 0.005], [0.05, 0.95]], "emissions": [0.04, 0.4]}
JSON
"${HMMIX[@]}" make_test_data -windows=10000 -chromosomes=2 -seed=7 -param=sim_params.json > log.make_test_data.txt

# Make the callability/mutation-rate inputs less trivial than the simulated ones:
# gaps, windows spanning boundaries, a weighted 4-column bed and non-uniform rates.
printf 'chr1\t0\t1234567\nchr1\t1300050\t9999999\nchr2\t500\t6500999\t0.5\nchr2\t6600000\t10000000\n' > weights_gaps.bed
awk 'BEGIN{OFS="\t"; for(c=1;c<=2;c++) for(s=0;s<10000000;s+=100000) print "chr"c, s, s+100000, 0.5+((s/100000)%7)/5}' > mutrates_var.bed

# 2) mutation rate from an "outgroup" (reuse obs positions)
"${HMMIX[@]}" mutation_rate -outgroup=obs.txt -weights=weights_gaps.bed -window_size=100000 -out=mutationrate.bed > log.mutation_rate.txt

# 3) train
"${HMMIX[@]}" train -obs=obs.txt -weights=weights.bed -mutrates=mutrates.bed -param=Initialguesses.json -out=trained.json > log.train.txt
"${HMMIX[@]}" train -obs=obs.txt -weights=weights_gaps.bed -mutrates=mutrates_var.bed -param=Initialguesses.json -out=trained_gaps.json > log.train_gaps.txt
"${HMMIX[@]}" train -obs=obs.txt -weights=weights.bed -mutrates=mutrates.bed -param=Initialguesses.json -haploid -out=trained_haploid.json > log.train_haploid.txt

# 4) decode (posterior / viterbi / hybrid, extra info, posterior table, chrom subset, haploid)
"${HMMIX[@]}" decode -obs=obs.txt -weights=weights.bed -mutrates=mutrates.bed -param=trained.json -out=decoded_post -posterior_probs=posterior_probs.txt > log.decode_post.txt
"${HMMIX[@]}" decode -obs=obs.txt -weights=weights_gaps.bed -mutrates=mutrates_var.bed -param=trained_gaps.json -out=decoded_gaps -extrainfo > log.decode_gaps.txt
"${HMMIX[@]}" decode -obs=obs.txt -weights=weights.bed -mutrates=mutrates.bed -param=trained.json -out=decoded_viterbi -viterbi > log.decode_viterbi.txt
"${HMMIX[@]}" decode -obs=obs.txt -weights=weights.bed -mutrates=mutrates.bed -param=trained.json -out=decoded_hybrid -hybrid=0.5 > log.decode_hybrid.txt
"${HMMIX[@]}" decode -obs=obs.txt -weights=weights.bed -mutrates=mutrates.bed -param=trained.json -out=decoded_chr2 -chrom=chr2 > log.decode_chr2.txt
"${HMMIX[@]}" decode -obs=obs.txt -weights=weights.bed -mutrates=mutrates.bed -param=trained_haploid.json -out=decoded_haploid -haploid -extrainfo > log.decode_haploid.txt

# 5) inhomogeneous Markov chain sampling
"${HMMIX[@]}" inhomogeneous -obs=obs.txt -weights=weights.bed -mutrates=mutrates.bed -param=trained.json -out=inhom -samples=3 -seed=3 -inhomogen_matrix=inhom_matrix.txt > log.inhomogeneous.txt

# 6) artemis (observed + simulated)
"${HMMIX[@]}" artemis -obs=obs.txt -weights=weights.bed -mutrates=mutrates.bed -param=trained.json -out=artemis.txt -out_plot=artemis.png -steps=11 > log.artemis.txt
"${HMMIX[@]}" artemis_sim -param=sim_params.json -out=artemis_sim.txt -out_plot=artemis_sim.png -n_sim_windows=20000 -steps=11 -seed=11 > log.artemis_sim.txt

rm -f artemis.png artemis_sim.png
