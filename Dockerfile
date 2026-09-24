# hmmix with bcftools and vcftools, which create_outgroup/create_ingroup and -admixpop need.
#
#   docker build -t hmmix .
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data hmmix make_test_data
#
# `docker build --target test .` also runs the test suite inside the image.
FROM python:3.13-slim AS base

RUN apt-get update \
 && apt-get install -y --no-install-recommends bcftools vcftools tabix \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
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

# Allow running as any user (docker run --user "$(id -u):$(id -g)"), so that output
# files in the mounted directory belong to the caller.
USER root
RUN chmod -R a+rwX /opt/numba-cache && rm -rf /tmp/matplotlib
USER hmmix

WORKDIR /data
ENTRYPOINT ["hmmix"]


# Runs the test suite; the image users get is the runtime stage below.
FROM base AS test
USER root
RUN pip install -c /opt/hmmix/docker/constraints.txt pytest
COPY tests /opt/hmmix/tests
RUN chown -R hmmix /opt/hmmix/tests
USER hmmix
WORKDIR /opt/hmmix
RUN python -m pytest -q tests


# Default target (last stage): the image without the tests.
FROM base AS runtime
