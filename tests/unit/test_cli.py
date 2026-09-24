"""Command line tests, including the subcommands that shell out to bcftools/vcftools."""

import shutil
import subprocess
import sys

import pytest

needs_bcftools = pytest.mark.skipif(
    shutil.which('bcftools') is None or shutil.which('vcftools') is None,
    reason='bcftools and vcftools are required',
)

# Five sites; out1/out2 form the outgroup, in1/in2 the ingroup.
#  100 A>G  polymorphic in outgroup                      -> outgroup, removed from ingroup
#  200 C>T  absent in outgroup, in1 het                   -> ingroup obs for in1
#  300 G>A  fixed alt in outgroup (derived count 0)       -> nowhere (in1 hom)
#  400 T>C  absent in outgroup, in1 hom, in2 het          -> ingroup obs for in2 only
#  500 A>T  polymorphic in outgroup                       -> outgroup
VCF = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tout1\tout2\tin1\tin2
chr1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t0/1\t0/0
chr1\t200\t.\tC\tT\t.\tPASS\t.\tGT\t0/0\t0/0\t0/1\t0/0
chr1\t300\t.\tG\tA\t.\tPASS\t.\tGT\t1/1\t1/1\t1/1\t1/1
chr1\t400\t.\tT\tC\t.\tPASS\t.\tGT\t0/0\t0/0\t1/1\t0/1
chr1\t500\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\t1/1\t0/0\t0/0
"""

# "Archaic" genomes used for -admixpop annotation
ARCHAIC_VCF = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tVindija\tDenisova
chr1\t200\t.\tC\tT\t.\tPASS\t.\tGT\t0/1\t0/0
chr1\t400\t.\tT\tC\t.\tPASS\t.\tGT\t1/1\t1/1
"""


def hmmix(*args, cwd):
    return subprocess.run([sys.executable, '-m', 'hmmix', *args], cwd=cwd, capture_output=True, text=True)


def data_lines(path):
    return [line.split('\t') for line in path.read_text().splitlines()[1:]]


def test_usage_without_arguments(tmp_path):
    result = hmmix(cwd=tmp_path)
    assert result.returncode == 0
    assert 'Script for identifying introgressed archaic segments' in result.stdout


def test_console_script_entry_point():
    from importlib.metadata import entry_points
    scripts = {ep.name: ep.value for ep in entry_points(group='console_scripts')}
    assert scripts['hmmix'] == 'hmmix.main:main'


def test_make_test_data_train_decode(tmp_path):
    assert hmmix('make_test_data', '-windows=2000', '-chromosomes=1', cwd=tmp_path).returncode == 0
    for name in ['obs.txt', 'weights.bed', 'mutrates.bed', 'Initialguesses.json']:
        assert (tmp_path / name).exists()
    result = hmmix('train', '-obs=obs.txt', '-weights=weights.bed', '-mutrates=mutrates.bed',
                   '-param=Initialguesses.json', '-out=trained.json', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    result = hmmix('decode', '-obs=obs.txt', '-param=trained.json', '-out=res/decoded', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    header, *rows = (tmp_path / 'res' / 'decoded.diploid.txt').read_text().splitlines()
    assert header.split('\t') == ['chrom', 'start', 'end', 'length', 'state', 'mean_prob', 'snps']
    assert rows


def test_decode_rejects_bad_hybrid_value(tmp_path):
    hmmix('make_test_data', '-windows=200', '-chromosomes=1', cwd=tmp_path)
    result = hmmix('decode', '-obs=obs.txt', '-hybrid=1.5', cwd=tmp_path)
    assert result.returncode != 0
    assert 'Hybrid parameter must be between 0 and 1' in result.stderr


@needs_bcftools
def test_create_outgroup_and_ingroup(tmp_path):
    (tmp_path / 'data.vcf').write_text(VCF)
    (tmp_path / 'ind.json').write_text('{"outgroup": ["out1", "out2"], "ingroup": ["in1", "in2"]}')

    result = hmmix('create_outgroup', '-ind=ind.json', '-vcf=data.vcf', '-out=outgroup.txt', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert data_lines(tmp_path / 'outgroup.txt') == [
        ['chr1', '100', 'A:3', 'G:1', 'A'],
        ['chr1', '500', 'A:1', 'T:3', 'T'],
    ]

    result = hmmix('create_ingroup', '-ind=ind.json', '-vcf=data.vcf', '-outgroup=outgroup.txt', '-out=obs', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert data_lines(tmp_path / 'obs.in1.txt') == [['chr1', '200', 'C', 'CT']]
    assert data_lines(tmp_path / 'obs.in2.txt') == [['chr1', '400', 'T', 'TC']]


@needs_bcftools
def test_decode_annotates_with_admixpop(tmp_path):
    (tmp_path / 'archaic.vcf').write_text(ARCHAIC_VCF)
    subprocess.run(['bcftools', 'view', '-Oz', '-o', 'archaic.vcf.gz', 'archaic.vcf'], cwd=tmp_path, check=True)
    subprocess.run(['bcftools', 'index', 'archaic.vcf.gz'], cwd=tmp_path, check=True)
    (tmp_path / 'obs.txt').write_text('chrom\tpos\tancestral_base\tgenotype\nchr1\t200\tC\tCT\nchr1\t400\tT\tTC\nchr1\t5500\tA\tAG\n')

    result = hmmix('decode', '-obs=obs.txt', '-admixpop=archaic.vcf.gz', '-extrainfo', '-out=dec', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    header, *rows = (tmp_path / 'dec.diploid.txt').read_text().splitlines()
    assert header.split('\t')[7:10] == ['admixpopvariants', 'Denisova', 'Vindija']
    # the one segment covers both SNPs: both are in Vindija, one in Denisova
    assert [r.split('\t')[7:10] for r in rows] == [['2', '1', '2']]
