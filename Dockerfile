# hmmix with bcftools and vcftools, which create_outgroup/create_ingroup and -admixpop need.
#
#   docker build -t hmmix .
#   docker run --rm -v "$PWD":/data hmmix make_test_data
#
# `docker build --target test .` also runs the test suite inside the image.
FROM python:3.13-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends bcftools vcftools tabix \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    NUMBA_CACHE_DIR=/opt/numba-cache

WORKDIR /opt/hmmix
COPY docker/constraints.txt docker/constraints.txt
COPY pyproject.toml README.md LICENSE ./
COPY src src
RUN pip install -c docker/constraints.txt .

RUN useradd --create-home hmmix && mkdir -p /opt/numba-cache && chown hmmix /opt/numba-cache
USER hmmix

# Compile the numba functions once at build time, so that `docker run` does not have to
# (numba recompiles by itself if the image runs on a CPU with different features).
COPY tests/regression/run_pipeline.sh /opt/hmmix/run_pipeline.sh
RUN bash /opt/hmmix/run_pipeline.sh /tmp/warmup hmmix && rm -rf /tmp/warmup

WORKDIR /data
ENTRYPOINT ["hmmix"]


FROM runtime AS test
USER root
RUN pip install -c /opt/hmmix/docker/constraints.txt pytest
COPY tests /opt/hmmix/tests
RUN chown -R hmmix /opt/hmmix/tests
USER hmmix
WORKDIR /opt/hmmix
RUN python -m pytest -q tests
