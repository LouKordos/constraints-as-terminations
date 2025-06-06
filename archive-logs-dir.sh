#!/usr/bin/bash
set -eux pipefail
TOTAL_BYTES=$(du -sb ./logs/clean_rl/ | cut -f1)
time tar -cf - ./logs/clean_rl/ | pv -s ${TOTAL_BYTES} | zstd --ultra -22 -T0 -o $HOME/Downloads/constraints-as-terminations-logs-and-runs.tar.zst

# To install: wget -qO- https://github.com/Blutsh/Swish/releases/download/1.0.1/swish-1.0.1-x86_64-unknown-linux-musl.tar.gz | tar xz --strip-components=1 -C ~/.local/bin/ --wildcards '*/swish'
swish $HOME/Downloads/constraints-as-terminations-logs-and-runs.tar.zst
